from math import inf, isclose, nextafter, pi, sin, cos

import pytest

from path_planner.v2.contracts import PlatformKindV2, PoseStateV2
from path_planner.v2.profiles import PlatformProfileV2, WheelKinematicSQPProfileV2
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    sample_wheel_segment_v2,
    wheel_relative_energy_v1,
    wheel_segment_jacobian_v2,
)


START = PoseStateV2(0.4, -0.3, 0.7)
PROFILE = WheelKinematicSQPProfileV2(
    profile=PlatformProfileV2(
        profile_id="scout-mini-wheel-kinematic-sqp/v1",
        platform_kind=PlatformKindV2.WHEEL,
        capability_revision="wheel_kinematic_corridor_sqp/v1",
        simulation_proxy=False,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=0.25,
        goal_heading_tolerance_rad=0.08726646259971647,
    )
)


def assert_pose_close(actual: PoseStateV2, expected: PoseStateV2, *, abs_tol: float) -> None:
    assert actual.x_m == pytest.approx(expected.x_m, abs=abs_tol)
    assert actual.y_m == pytest.approx(expected.y_m, abs=abs_tol)
    assert actual.heading_rad == pytest.approx(expected.heading_rad, abs=abs_tol)


@pytest.mark.parametrize(
    ("v", "omega", "dt", "expected"),
    [
        (1.0, 0.0, 2.0, PoseStateV2(2.0, 0.0, 0.0)),
        (-1.0, 0.0, 2.0, PoseStateV2(-2.0, 0.0, 0.0)),
        (0.0, pi / 2.0, 1.0, PoseStateV2(0.0, 0.0, pi / 2.0)),
        (0.0, 0.0, 0.05, PoseStateV2(0.0, 0.0, 0.0)),
    ],
)
def test_analytic_wheel_integration_closed_forms(v, omega, dt, expected) -> None:
    actual = integrate_wheel_segment_v2(PoseStateV2(0.0, 0.0, 0.0), v, omega, dt)
    assert_pose_close(actual, expected, abs_tol=1.0e-14)


def test_arc_uses_the_exact_sinc_form() -> None:
    end = integrate_wheel_segment_v2(PoseStateV2(1.0, 2.0, 0.3), 0.7, 0.2, 1.5)
    radius = 0.7 / 0.2
    assert end.x_m == pytest.approx(1.0 + radius * (sin(0.6) - sin(0.3)), abs=1e-14)
    assert end.y_m == pytest.approx(2.0 - radius * (cos(0.6) - cos(0.3)), abs=1e-14)
    assert end.heading_rad == pytest.approx(0.6, abs=1e-14)


def test_sinc_branch_is_continuous_at_exact_half_angle_threshold() -> None:
    threshold_omega = 2.0e-4
    below = integrate_wheel_segment_v2(START, 0.8, nextafter(threshold_omega, 0.0), 1.0)
    at = integrate_wheel_segment_v2(START, 0.8, threshold_omega, 1.0)
    above = integrate_wheel_segment_v2(START, 0.8, nextafter(threshold_omega, inf), 1.0)
    assert_pose_close(below, at, abs_tol=2.0e-15)
    assert_pose_close(at, above, abs_tol=2.0e-15)


def _finite_difference_column(
    values: tuple[float, float, float, float, float, float], index: int
) -> tuple[float, float, float]:
    step = 1.0e-6
    lower = list(values)
    upper = list(values)
    lower[index] -= step
    upper[index] += step
    left = integrate_wheel_segment_v2(PoseStateV2(*lower[:3]), *lower[3:])
    right = integrate_wheel_segment_v2(PoseStateV2(*upper[:3]), *upper[3:])
    return tuple(
        (high - low) / (2.0 * step)
        for low, high in zip(
            (left.x_m, left.y_m, left.heading_rad),
            (right.x_m, right.y_m, right.heading_rad),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "values",
    [
        (0.4, -0.3, 0.7, 0.8, 0.0, 1.2),
        (0.4, -0.3, 0.7, 0.8, 1.0e-9, 1.2),
        (0.4, -0.3, 0.7, 0.8, 0.3, 1.2),
        (0.4, -0.3, 0.7, -0.8, -0.2, 1.2),
        (0.4, -0.3, 0.7, 0.0, 0.5, 1.2),
    ],
)
def test_analytic_six_column_jacobian_matches_central_difference(values) -> None:
    actual = wheel_segment_jacobian_v2(PoseStateV2(*values[:3]), *values[3:])
    assert len(actual) == 3
    assert all(len(row) == 6 for row in actual)
    for column in range(6):
        expected = _finite_difference_column(values, column)
        for row in range(3):
            assert isclose(actual[row][column], expected[row], rel_tol=2e-6, abs_tol=2e-8)


def test_sampling_uses_exact_translation_and_heading_ceilings() -> None:
    samples = sample_wheel_segment_v2(START, 0.6, 0.2, 1.0, PROFILE)
    assert len(samples) == 4
    assert samples[0] is START
    assert samples[-1] == integrate_wheel_segment_v2(START, 0.6, 0.2, 1.0)
    assert samples[1] == integrate_wheel_segment_v2(START, 0.6, 0.2, 1.0 / 3.0)


def test_relative_energy_has_one_shared_analytic_authority() -> None:
    assert wheel_relative_energy_v1(-0.5, 0.2, 2.0, PROFILE) == pytest.approx(1.43, abs=1e-15)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (integrate_wheel_segment_v2, (START, float("nan"), 0.0, 1.0)),
        (integrate_wheel_segment_v2, (START, 1.0, 0.0, 0.0)),
        (sample_wheel_segment_v2, (START, 1.0, 0.0, 0.0, PROFILE)),
    ],
)
def test_kinematic_authorities_reject_nonfinite_or_nonpositive_inputs(function, args) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(*args)
