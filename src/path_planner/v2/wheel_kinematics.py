from __future__ import annotations

from math import ceil, cos, isfinite, sin
from numbers import Real

from path_planner.v2.contracts import PoseStateV2
from path_planner.v2.profiles import WheelKinematicSQPProfileV2


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive(value: object, name: str) -> float:
    normalized = _finite(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _exact_start(start: object) -> PoseStateV2:
    if type(start) is not PoseStateV2:
        raise TypeError("start must be exact PoseStateV2")
    return start


def _exact_profile(profile: object) -> WheelKinematicSQPProfileV2:
    if type(profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
    return profile


def _sinc_v1(h: float) -> float:
    if abs(h) <= 1.0e-4:
        h2 = h * h
        return 1.0 - h2 / 6.0 + h2 * h2 / 120.0 - h2 * h2 * h2 / 5040.0
    return sin(h) / h


def _sinc_prime_v1(h: float) -> float:
    if abs(h) <= 1.0e-4:
        h2 = h * h
        return -h / 3.0 + h * h2 / 30.0 - h * h2 * h2 / 840.0
    return (h * cos(h) - sin(h)) / (h * h)


def integrate_wheel_segment_v2(
    start: PoseStateV2,
    v_mps: float,
    omega_radps: float,
    duration_s: float,
) -> PoseStateV2:
    start = _exact_start(start)
    v = _finite(v_mps, "v_mps")
    omega = _finite(omega_radps, "omega_radps")
    dt = _positive(duration_s, "duration_s")
    h = 0.5 * omega * dt
    travel = v * dt * _sinc_v1(h)
    midpoint_heading = start.heading_rad + h
    return PoseStateV2(
        start.x_m + travel * cos(midpoint_heading),
        start.y_m + travel * sin(midpoint_heading),
        start.heading_rad + 2.0 * h,
    )


def wheel_segment_jacobian_v2(
    start: PoseStateV2,
    v_mps: float,
    omega_radps: float,
    duration_s: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    start = _exact_start(start)
    v = _finite(v_mps, "v_mps")
    omega = _finite(omega_radps, "omega_radps")
    dt = _positive(duration_s, "duration_s")
    h = 0.5 * omega * dt
    sinc = _sinc_v1(h)
    sinc_prime = _sinc_prime_v1(h)
    travel = v * dt * sinc
    heading = start.heading_rad + h
    heading_cos = cos(heading)
    heading_sin = sin(heading)
    travel_v = dt * sinc
    travel_omega = 0.5 * v * dt * dt * sinc_prime
    travel_dt = v * sinc + 0.5 * v * dt * omega * sinc_prime
    return (
        (
            1.0,
            0.0,
            -travel * heading_sin,
            travel_v * heading_cos,
            travel_omega * heading_cos - 0.5 * dt * travel * heading_sin,
            travel_dt * heading_cos - 0.5 * omega * travel * heading_sin,
        ),
        (
            0.0,
            1.0,
            travel * heading_cos,
            travel_v * heading_sin,
            travel_omega * heading_sin + 0.5 * dt * travel * heading_cos,
            travel_dt * heading_sin + 0.5 * omega * travel * heading_cos,
        ),
        (0.0, 0.0, 1.0, 0.0, dt, omega),
    )


def sample_wheel_segment_v2(
    start: PoseStateV2,
    v_mps: float,
    omega_radps: float,
    duration_s: float,
    profile: WheelKinematicSQPProfileV2,
) -> tuple[PoseStateV2, ...]:
    start = _exact_start(start)
    v = _finite(v_mps, "v_mps")
    omega = _finite(omega_radps, "omega_radps")
    dt = _positive(duration_s, "duration_s")
    profile = _exact_profile(profile)
    count = max(
        1,
        ceil(abs(v) * dt / profile.observation_sample_translation_m),
        ceil(abs(omega) * dt / profile.observation_sample_heading_rad),
    )
    return (start,) + tuple(
        integrate_wheel_segment_v2(start, v, omega, dt * index / count)
        for index in range(1, count + 1)
    )


def wheel_relative_energy_v1(
    v_mps: float,
    omega_radps: float,
    duration_s: float,
    profile: WheelKinematicSQPProfileV2,
) -> float:
    v = _finite(v_mps, "v_mps")
    omega = _finite(omega_radps, "omega_radps")
    dt = _positive(duration_s, "duration_s")
    profile = _exact_profile(profile)
    translation = abs(v) * dt * profile.translation_energy_per_m
    if v < 0.0:
        translation *= profile.reverse_energy_multiplier
    rotation = abs(omega) * dt * profile.rotation_energy_per_rad
    idle = dt * profile.idle_energy_per_s
    return (translation + rotation + idle) / profile.energy_normalization
