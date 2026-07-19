from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, isfinite, pi, sin
from numbers import Real

from path_planner.core.models import Cell


MAX_BALLISTIC_SAMPLES_V2 = 100_000


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
    if type(value.x) is not int or type(value.y) is not int:
        raise TypeError("cell coordinates must be exact int values")
    return Cell(value.x, value.y)


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
    audited_start = BallisticStartV2(start.x_m, start.y_m, start.z_m)
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

    samples = [
        BallisticSampleV2(
            0.0,
            audited_start.x_m,
            audited_start.y_m,
            audited_start.z_m,
        )
    ]
    for index in range(1, interval_count):
        time_s = _derived_finite(float(index) * dt, "sample time_s")
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
