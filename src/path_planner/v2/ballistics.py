from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, erfc, erf, floor, fsum, isfinite, pi, sin, sqrt
from numbers import Real

from path_planner.core.models import Cell, WorldPoint
from path_planner.v2.terrain import FineGridGeometryV2


MAX_BALLISTIC_SAMPLES_V2 = 100_000
MAX_LANDING_ZONE_CANDIDATES_V2 = 1_000_000
_SQRT_TWO = sqrt(2.0)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, RuntimeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def _positive_real(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _derived_finite(value: float, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return 0.0 if value == 0.0 else value


def _exact_cell(value: object) -> Cell:
    if type(value) is not Cell:
        raise TypeError("cell must be exact Cell")
    try:
        x, y = value.x, value.y
    except AttributeError:
        raise TypeError("cell must have exact fields") from None
    if type(x) is not int or type(y) is not int:
        raise TypeError("cell coordinates must be exact int values")
    return Cell(x, y)


@dataclass(frozen=True, slots=True)
class BallisticStartV2:
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class BallisticSampleV2:
    time_s: float
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite_real(self.time_s, "time_s"))
        if self.time_s < 0.0:
            raise ValueError("time_s must be nonnegative")
        for name in ("x_m", "y_m", "z_m"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class LandingCellMassV2:
    cell: Cell
    probability_mass: float
    in_bounds: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _exact_cell(self.cell))
        mass = _finite_real(self.probability_mass, "probability_mass")
        if not 0.0 < mass <= 1.0:
            raise ValueError("probability_mass must be in (0, 1]")
        object.__setattr__(self, "probability_mass", mass)
        if type(self.in_bounds) is not bool:
            raise TypeError("in_bounds must be exact bool")


def sample_ballistic_arc(
    start: BallisticStartV2,
    speed_mps: float,
    elevation_rad: float,
    azimuth_rad: float,
    g_mps2: float,
    dt_s: float,
) -> tuple[BallisticSampleV2, ...]:
    if type(start) is not BallisticStartV2:
        raise TypeError("start must be exact BallisticStartV2")
    try:
        start_x_m, start_y_m, start_z_m = start.x_m, start.y_m, start.z_m
    except AttributeError:
        raise TypeError("start must have exact fields") from None
    audited_start = BallisticStartV2(start_x_m, start_y_m, start_z_m)
    speed = _positive_real(speed_mps, "speed_mps")
    elevation = _finite_real(elevation_rad, "elevation_rad")
    azimuth = _finite_real(azimuth_rad, "azimuth_rad")
    gravity = _positive_real(g_mps2, "g_mps2")
    dt = _positive_real(dt_s, "dt_s")
    if not 0.0 < elevation < pi / 2.0:
        raise ValueError("elevation_rad must be in (0, pi/2)")

    horizontal_speed = _derived_finite(speed * cos(elevation), "horizontal speed")
    vertical_speed = _derived_finite(speed * sin(elevation), "vertical speed")
    vx = _derived_finite(horizontal_speed * cos(azimuth), "x velocity")
    vy = _derived_finite(horizontal_speed * sin(azimuth), "y velocity")
    vertical_time_scale = _derived_finite(
        vertical_speed / gravity,
        "vertical speed / gravity",
    )
    flight_time = _derived_finite(2.0 * vertical_time_scale, "flight time")
    if flight_time <= 0.0:
        raise ValueError("flight time must be positive")
    interval_ratio = _derived_finite(flight_time / dt, "flight time / dt_s")
    if interval_ratio <= 0.0:
        raise ValueError("flight time / dt_s must be positive")
    if interval_ratio > float(MAX_BALLISTIC_SAMPLES_V2 - 1):
        raise ValueError(
            f"sample_count must not exceed {MAX_BALLISTIC_SAMPLES_V2}"
        )

    horizontal_dx = _derived_finite(vx * flight_time, "landing x displacement")
    horizontal_dy = _derived_finite(vy * flight_time, "landing y displacement")
    landing_x = _derived_finite(audited_start.x_m + horizontal_dx, "landing x_m")
    landing_y = _derived_finite(audited_start.y_m + horizontal_dy, "landing y_m")
    apex_height = _derived_finite(
        vertical_time_scale * (0.5 * vertical_speed),
        "apex height",
    )
    _derived_finite(audited_start.z_m + apex_height, "apex z_m")

    try:
        interval_count = ceil(interval_ratio)
    except (OverflowError, ValueError):
        raise ValueError("sample_count must be finite") from None
    sample_count = interval_count + 1
    if sample_count > MAX_BALLISTIC_SAMPLES_V2:
        raise ValueError(
            f"sample_count must not exceed {MAX_BALLISTIC_SAMPLES_V2}"
        )

    interior_times = tuple(
        _derived_finite(float(index) * dt, "sample time_s")
        for index in range(1, interval_count)
    )
    scheduled_times = (0.0, *interior_times, flight_time)
    if any(
        right - left > dt
        for left, right in zip(scheduled_times, scheduled_times[1:])
    ):
        interval_count += 1
        sample_count = interval_count + 1
        if sample_count > MAX_BALLISTIC_SAMPLES_V2:
            raise ValueError(
                f"sample_count must not exceed {MAX_BALLISTIC_SAMPLES_V2}"
            )
        interior_times = tuple(
            _derived_finite(
                flight_time
                * _derived_finite(
                    float(index) / float(interval_count),
                    "sample time fraction",
                ),
                "sample time_s",
            )
            for index in range(1, interval_count)
        )
        scheduled_times = (0.0, *interior_times, flight_time)
        if any(
            right - left > dt
            for left, right in zip(scheduled_times, scheduled_times[1:])
        ):
            raise ValueError("sample intervals must not exceed dt_s")

    samples = [
        BallisticSampleV2(
            0.0,
            audited_start.x_m,
            audited_start.y_m,
            audited_start.z_m,
        )
    ]
    for time_s in interior_times:
        if not samples[-1].time_s < time_s < flight_time:
            raise ValueError("sample times must remain strictly interior")
        fraction = _derived_finite(time_s / flight_time, "sample time fraction")
        x_m = _derived_finite(
            audited_start.x_m + horizontal_dx * fraction,
            "sample x_m",
        )
        y_m = _derived_finite(
            audited_start.y_m + horizontal_dy * fraction,
            "sample y_m",
        )
        z_m = _derived_finite(
            audited_start.z_m
            + 4.0 * apex_height * fraction * (1.0 - fraction),
            "sample z_m",
        )
        samples.append(BallisticSampleV2(time_s, x_m, y_m, z_m))

    samples.append(
        BallisticSampleV2(
            flight_time,
            landing_x,
            landing_y,
            audited_start.z_m,
        )
    )
    return tuple(samples)


def normal_interval_mass(lo: float, hi: float, mean: float, sigma: float) -> float:
    lower = _finite_real(lo, "lo")
    upper = _finite_real(hi, "hi")
    center = _finite_real(mean, "mean")
    scale = _positive_real(sigma, "sigma")
    if lower > upper:
        raise ValueError("lo must be less than or equal to hi")
    if lower == upper:
        return 0.0
    lower_delta = _derived_finite(lower - center, "lo - mean")
    upper_delta = _derived_finite(upper - center, "hi - mean")
    lower_z = _derived_finite(lower_delta / scale, "standardized lo")
    upper_z = _derived_finite(upper_delta / scale, "standardized hi")
    if lower_z >= 0.0:
        mass = 0.5 * (
            erfc(lower_z / _SQRT_TWO) - erfc(upper_z / _SQRT_TWO)
        )
    elif upper_z <= 0.0:
        mass = 0.5 * (
            erfc(-upper_z / _SQRT_TWO) - erfc(-lower_z / _SQRT_TWO)
        )
    else:
        mass = 0.5 * (
            erf(upper_z / _SQRT_TWO) - erf(lower_z / _SQRT_TWO)
        )
    mass = _derived_finite(mass, "normal interval mass")
    return min(1.0, max(0.0, mass))


def _audited_world_point(value: object) -> WorldPoint:
    if type(value) is not WorldPoint:
        raise TypeError("mean_xy must be exact WorldPoint")
    try:
        x, y = value.x, value.y
    except AttributeError:
        raise TypeError("mean_xy must have exact fields") from None
    return WorldPoint(
        _finite_real(x, "mean_xy.x"),
        _finite_real(y, "mean_xy.y"),
    )


def _audited_geometry(value: object) -> FineGridGeometryV2:
    if type(value) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    try:
        width = value.width
        height = value.height
        raw_origin = value.origin
        resolution_m = value.resolution_m
        frame_id = value.frame_id
    except AttributeError:
        raise TypeError("geometry must have exact fields") from None
    if type(width) is not int or width <= 0:
        raise TypeError("geometry.width must be an exact int greater than zero")
    if type(height) is not int or height <= 0:
        raise TypeError("geometry.height must be an exact int greater than zero")
    if type(raw_origin) is not tuple or len(raw_origin) != 2:
        raise TypeError("geometry.origin must be an exact two-item tuple")
    origin = (
        _finite_real(raw_origin[0], "geometry.origin[0]"),
        _finite_real(raw_origin[1], "geometry.origin[1]"),
    )
    if type(resolution_m) is not float:
        raise TypeError("geometry.resolution_m must be exact float")
    if resolution_m != 0.5:
        raise ValueError("geometry.resolution_m must be exactly 0.5")
    if type(frame_id) is not str or not frame_id.strip():
        raise TypeError("geometry.frame_id must be exact nonempty str")
    return FineGridGeometryV2(
        width=width,
        height=height,
        origin=origin,
        frame_id=frame_id,
        resolution_m=resolution_m,
    )


def _axis_boundary(origin: float, index: int, resolution: float, name: str) -> float:
    try:
        offset = float(index) * resolution
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    return _derived_finite(origin + offset, name)


def _axis_interval_mass(
    index: int,
    origin: float,
    resolution: float,
    mean: float,
    sigma: float,
    axis: str,
) -> float:
    lower = _axis_boundary(origin, index, resolution, f"{axis} lower boundary")
    upper = _axis_boundary(origin, index + 1, resolution, f"{axis} upper boundary")
    if upper <= lower:
        raise ValueError(f"{axis} cell boundaries must remain representable")
    return normal_interval_mass(lower, upper, mean, sigma)


def _left_tail(boundary: float, mean: float, sigma: float, name: str) -> float:
    z = _derived_finite((boundary - mean) / sigma, name)
    return 0.5 * erfc(-z / _SQRT_TWO)


def _right_tail(boundary: float, mean: float, sigma: float, name: str) -> float:
    z = _derived_finite((boundary - mean) / sigma, name)
    return 0.5 * erfc(z / _SQRT_TWO)


def _shortest_prefix_length(masses: tuple[float, ...], threshold: float) -> int | None:
    if not masses or fsum(masses) < threshold:
        return None
    lower, upper = 1, len(masses)
    while lower < upper:
        middle = (lower + upper) // 2
        if fsum(masses[:middle]) >= threshold:
            upper = middle
        else:
            lower = middle + 1
    return lower


def landing_zone_cells(
    mean_xy: WorldPoint,
    sigma_m: float,
    probability_threshold: float,
    geometry: FineGridGeometryV2,
) -> tuple[LandingCellMassV2, ...]:
    mean = _audited_world_point(mean_xy)
    audited_geometry = _audited_geometry(geometry)
    sigma = _positive_real(sigma_m, "sigma_m")
    threshold = _finite_real(probability_threshold, "probability_threshold")
    if not 0.0 < threshold < 1.0:
        raise ValueError("probability_threshold must be in (0, 1)")
    cap = MAX_LANDING_ZONE_CANDIDATES_V2
    if type(cap) is not int or cap <= 0:
        raise ValueError("candidate cap must be an exact positive int")

    resolution = audited_geometry.resolution_m
    x_coordinate = _derived_finite(
        (mean.x - audited_geometry.origin[0]) / resolution,
        "mean x grid coordinate",
    )
    y_coordinate = _derived_finite(
        (mean.y - audited_geometry.origin[1]) / resolution,
        "mean y grid coordinate",
    )
    try:
        center_x, center_y = floor(x_coordinate), floor(y_coordinate)
    except (OverflowError, ValueError):
        raise ValueError("mean grid coordinates must be representable") from None

    max_x_mass = _axis_interval_mass(
        center_x, audited_geometry.origin[0], resolution, mean.x, sigma, "x"
    )
    max_y_mass = _axis_interval_mass(
        center_y, audited_geometry.origin[1], resolution, mean.y, sigma, "y"
    )
    evaluated: list[tuple[float, float, int, int, LandingCellMassV2]] = []
    radius = 0
    while True:
        side = 2 * radius + 1
        evaluated_count = side * side
        if evaluated_count > cap:
            raise ValueError(f"landing-zone candidate count exceeds candidate cap {cap}")

        for y_index in range(center_y - radius, center_y + radius + 1):
            for x_index in range(center_x - radius, center_x + radius + 1):
                if radius > 0 and max(
                    abs(x_index - center_x), abs(y_index - center_y)
                ) != radius:
                    continue
                x_mass = _axis_interval_mass(
                    x_index,
                    audited_geometry.origin[0],
                    resolution,
                    mean.x,
                    sigma,
                    "x",
                )
                y_mass = _axis_interval_mass(
                    y_index,
                    audited_geometry.origin[1],
                    resolution,
                    mean.y,
                    sigma,
                    "y",
                )
                mass = _derived_finite(x_mass * y_mass, "cell probability mass")
                x_lower = _axis_boundary(
                    audited_geometry.origin[0], x_index, resolution, "x lower boundary"
                )
                y_lower = _axis_boundary(
                    audited_geometry.origin[1], y_index, resolution, "y lower boundary"
                )
                center_world_x = _derived_finite(
                    x_lower + 0.5 * resolution, "cell center x"
                )
                center_world_y = _derived_finite(
                    y_lower + 0.5 * resolution, "cell center y"
                )
                dx = _derived_finite(center_world_x - mean.x, "cell center dx")
                dy = _derived_finite(center_world_y - mean.y, "cell center dy")
                distance_sq = _derived_finite(dx * dx + dy * dy, "cell distance squared")
                if mass > 0.0:
                    cell = Cell(x_index, y_index)
                    item = LandingCellMassV2(
                        cell=cell,
                        probability_mass=mass,
                        in_bounds=(
                            0 <= x_index < audited_geometry.width
                            and 0 <= y_index < audited_geometry.height
                        ),
                    )
                    evaluated.append((mass, distance_sq, y_index, x_index, item))

        evaluated.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        masses = tuple(row[0] for row in evaluated)
        prefix_length = _shortest_prefix_length(masses, threshold)
        if prefix_length is not None:
            prefix = tuple(row[4] for row in evaluated[:prefix_length])
            last_mass = prefix[-1].probability_mass
            x_left = _axis_boundary(
                audited_geometry.origin[0], center_x - radius, resolution, "x tail left"
            )
            x_right = _axis_boundary(
                audited_geometry.origin[0], center_x + radius + 1, resolution, "x tail right"
            )
            y_bottom = _axis_boundary(
                audited_geometry.origin[1], center_y - radius, resolution, "y tail bottom"
            )
            y_top = _axis_boundary(
                audited_geometry.origin[1], center_y + radius + 1, resolution, "y tail top"
            )
            unseen_mass_upper_bound = max(
                _left_tail(x_left, mean.x, sigma, "left x tail") * max_y_mass,
                _right_tail(x_right, mean.x, sigma, "right x tail") * max_y_mass,
                _left_tail(y_bottom, mean.y, sigma, "lower y tail") * max_x_mass,
                _right_tail(y_top, mean.y, sigma, "upper y tail") * max_x_mass,
            )
            if unseen_mass_upper_bound < last_mass:
                if fsum(item.probability_mass for item in prefix[:-1]) >= threshold:
                    raise RuntimeError("landing-zone prefix is not shortest")
                return prefix
        radius += 1
