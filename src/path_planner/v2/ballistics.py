from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import (
    cos,
    erfc,
    erf,
    floor,
    fsum,
    isfinite,
    isqrt,
    nextafter,
    pi,
    sin,
    sqrt,
    ulp,
)
from numbers import Real

from path_planner.core.models import Cell, WorldPoint
from path_planner.v2.terrain import FineGridGeometryV2


MAX_BALLISTIC_SAMPLES_V2 = 100_000
MAX_LANDING_ZONE_CANDIDATES_V2 = 1_000_000
_MAX_SAMPLE_TIME_ULP_CORRECTIONS = 4
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


def _derived_positive(value: float, name: str) -> float:
    normalized = _derived_finite(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} underflowed and is not representable")
    return normalized


def _add_representable_offset(origin: float, offset: float, name: str) -> float:
    result = _derived_finite(origin + offset, name)
    if offset != 0.0 and result == origin:
        raise ValueError(
            f"{name} failed representability because a nonzero offset was absorbed"
        )
    return result


def _exact_positive_ratio_ceil(numerator: float, denominator: float) -> int:
    numerator_integer, numerator_denominator = numerator.as_integer_ratio()
    denominator_integer, denominator_denominator = denominator.as_integer_ratio()
    quotient, remainder = divmod(
        numerator_integer * denominator_denominator,
        numerator_denominator * denominator_integer,
    )
    return quotient + int(remainder != 0)


def _interior_sample_times(
    flight_time: float,
    dt_s: float,
    interval_count: int,
    max_sample_count: int,
) -> tuple[float, ...]:
    exact_sample_lower_bound = interval_count + 1
    omits_endpoint_collision = False
    if interval_count > 1:
        last_nominal = _derived_finite(
            float(interval_count - 1) * dt_s,
            "sample time_s",
        )
        omits_endpoint_collision = (
            abs(last_nominal - flight_time)
            <= _MAX_SAMPLE_TIME_ULP_CORRECTIONS * ulp(last_nominal)
        )
    # The exact endpoint owns a colliding final anchor, so that omitted
    # interior slot remains available to one bounded repair.
    repair_budget = (
        max_sample_count
        - exact_sample_lower_bound
        + int(omits_endpoint_collision)
    )
    times: list[float] = []
    previous = 0.0
    for index in range(1, interval_count):
        nominal = _derived_finite(float(index) * dt_s, "sample time_s")
        if omits_endpoint_collision and index == interval_count - 1:
            continue
        candidate = nominal
        for correction_count in range(_MAX_SAMPLE_TIME_ULP_CORRECTIONS + 1):
            if (
                previous < candidate < flight_time
                and candidate - previous <= dt_s
            ):
                break
            if correction_count < _MAX_SAMPLE_TIME_ULP_CORRECTIONS:
                candidate = nextafter(candidate, float("-inf"))
        else:
            if not nominal < flight_time:
                raise ValueError("sample time_s failed representability")
            repair = _bounded_local_repair(previous, nominal, dt_s)
            if repair is None:
                raise ValueError("sample time_s failed representability")
            if repair_budget <= 0:
                raise ValueError(
                    f"sample_count must not exceed {max_sample_count}"
                )
            times.append(repair)
            repair_budget -= 1
            previous = repair
            candidate = nominal
            if not (
                previous < candidate < flight_time
                and candidate - previous <= dt_s
            ):
                raise ValueError("sample time_s failed representability")
        times.append(candidate)
        previous = candidate

    if (
        flight_time - previous > dt_s
        or len(times) + 2 < exact_sample_lower_bound
    ):
        repair = _bounded_local_repair(previous, flight_time, dt_s)
        if repair is None:
            raise ValueError("sample time_s failed representability")
        if repair_budget <= 0:
            raise ValueError(f"sample_count must not exceed {max_sample_count}")
        times.append(repair)
        repair_budget -= 1
        previous = repair
    if not previous < flight_time or flight_time - previous > dt_s:
        raise ValueError("sample time_s failed representability")
    actual_sample_count = len(times) + 2
    if actual_sample_count < exact_sample_lower_bound:
        raise ValueError("sample_count must satisfy exact interval lower bound")
    if actual_sample_count > max_sample_count:
        raise ValueError(f"sample_count must not exceed {max_sample_count}")
    return tuple(times)


def _bounded_local_repair(
    previous: float,
    target: float,
    dt_s: float,
) -> float | None:
    """Find one bridge around step/endpoint bounds with fixed four-ULP searches."""
    centers = (
        previous + dt_s,
        target - dt_s,
        nextafter(previous, float("inf")),
        nextafter(target, float("-inf")),
    )
    for raw_center in centers:
        if not isfinite(raw_center):
            continue
        center = _derived_finite(raw_center, "sample repair time_s")
        candidates = [center]
        lower = center
        upper = center
        for _ in range(_MAX_SAMPLE_TIME_ULP_CORRECTIONS):
            lower = nextafter(lower, float("-inf"))
            upper = nextafter(upper, float("inf"))
            candidates.extend((lower, upper))
        for candidate in candidates:
            if (
                previous < candidate < target
                and candidate - previous <= dt_s
                and target - candidate <= dt_s
            ):
                return candidate
    return None


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

    horizontal_speed = _derived_positive(speed * cos(elevation), "horizontal speed")
    vertical_speed = _derived_positive(speed * sin(elevation), "vertical speed")
    x_direction = _derived_finite(cos(azimuth), "cos(azimuth_rad)")
    y_direction = _derived_finite(sin(azimuth), "sin(azimuth_rad)")
    vx = _derived_finite(horizontal_speed * x_direction, "x velocity")
    vy = _derived_finite(horizontal_speed * y_direction, "y velocity")
    if x_direction != 0.0 and vx == 0.0:
        raise ValueError("x velocity failed representability")
    if y_direction != 0.0 and vy == 0.0:
        raise ValueError("y velocity failed representability")
    vertical_time_scale = _derived_positive(
        vertical_speed / gravity,
        "vertical speed / gravity",
    )
    flight_time = _derived_positive(2.0 * vertical_time_scale, "flight time")

    horizontal_dx = _derived_finite(vx * flight_time, "landing x displacement")
    horizontal_dy = _derived_finite(vy * flight_time, "landing y displacement")
    if vx != 0.0 and horizontal_dx == 0.0:
        raise ValueError("landing x displacement failed representability")
    if vy != 0.0 and horizontal_dy == 0.0:
        raise ValueError("landing y displacement failed representability")
    _derived_positive(horizontal_speed * flight_time, "horizontal range")
    landing_x = _add_representable_offset(
        audited_start.x_m, horizontal_dx, "landing x_m"
    )
    landing_y = _add_representable_offset(
        audited_start.y_m, horizontal_dy, "landing y_m"
    )
    apex_height = _derived_positive(
        vertical_time_scale * (0.5 * vertical_speed),
        "apex height",
    )
    _add_representable_offset(audited_start.z_m, apex_height, "apex z_m")

    interval_count = _exact_positive_ratio_ceil(flight_time, dt)
    sample_count = interval_count + 1
    if sample_count > MAX_BALLISTIC_SAMPLES_V2:
        raise ValueError(
            f"sample_count must not exceed {MAX_BALLISTIC_SAMPLES_V2}"
        )

    interior_times = _interior_sample_times(
        flight_time,
        dt,
        interval_count,
        MAX_BALLISTIC_SAMPLES_V2,
    )

    samples = [
        BallisticSampleV2(
            0.0,
            audited_start.x_m,
            audited_start.y_m,
            audited_start.z_m,
        )
    ]
    for time_s in interior_times:
        fraction = _derived_positive(time_s / flight_time, "sample time fraction")
        if fraction >= 1.0:
            raise ValueError("sample time fraction failed representability")
        x_offset = _derived_finite(
            horizontal_dx * fraction, "sample x displacement"
        )
        y_offset = _derived_finite(
            horizontal_dy * fraction, "sample y displacement"
        )
        z_offset = _derived_positive(
            4.0 * apex_height * fraction * (1.0 - fraction),
            "sample z displacement",
        )
        if horizontal_dx != 0.0 and x_offset == 0.0:
            raise ValueError("sample x_m failed representability")
        if horizontal_dy != 0.0 and y_offset == 0.0:
            raise ValueError("sample y_m failed representability")
        x_m = _add_representable_offset(
            audited_start.x_m, x_offset, "sample x_m"
        )
        y_m = _add_representable_offset(
            audited_start.y_m, y_offset, "sample y_m"
        )
        z_m = _add_representable_offset(
            audited_start.z_m, z_offset, "sample z_m"
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


def _axis_cell_center(
    origin: float,
    index: int,
    resolution: float,
    axis: str,
) -> float:
    lower = _axis_boundary(origin, index, resolution, f"{axis} lower boundary")
    upper = _axis_boundary(origin, index + 1, resolution, f"{axis} upper boundary")
    try:
        center = origin + (index + 0.5) * resolution
    except OverflowError:
        raise ValueError(f"{axis} cell center must be representable") from None
    if not isfinite(center) or not lower < center < upper:
        raise ValueError(f"{axis} cell center must be representable")
    return 0.0 if center == 0.0 else center


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


def _iter_square_perimeter_cells(
    center_x: int,
    center_y: int,
    radius: int,
) -> Iterator[tuple[int, int]]:
    if radius == 0:
        yield center_x, center_y
        return

    left = center_x - radius
    right = center_x + radius
    top = center_y - radius
    bottom = center_y + radius
    for x_index in range(left, right + 1):
        yield x_index, top
    for y_index in range(top + 1, bottom + 1):
        yield right, y_index
    for x_index in range(right - 1, left - 1, -1):
        yield x_index, bottom
    for y_index in range(bottom - 1, top, -1):
        yield left, y_index


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
    final_radius = (isqrt(cap) - 1) // 2

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
    prefix_evaluation_radius = 0
    while True:
        side = 2 * radius + 1
        evaluated_count = side * side
        if evaluated_count > cap:
            raise ValueError(f"landing-zone candidate count exceeds candidate cap {cap}")

        for x_index, y_index in _iter_square_perimeter_cells(
            center_x, center_y, radius
        ):
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
            center_world_x = _axis_cell_center(
                audited_geometry.origin[0], x_index, resolution, "x"
            )
            center_world_y = _axis_cell_center(
                audited_geometry.origin[1], y_index, resolution, "y"
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

        if radius != prefix_evaluation_radius and radius != final_radius:
            radius += 1
            continue

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
        if radius == prefix_evaluation_radius:
            prefix_evaluation_radius = 2 * radius + 1
        radius += 1
