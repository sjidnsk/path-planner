from dataclasses import FrozenInstanceError, fields
from math import nextafter, pi, sin

import pytest

from path_planner.core.models import Cell
from path_planner.v2.ballistics import (
    MAX_BALLISTIC_SAMPLES_V2,
    BallisticSampleV2,
    BallisticStartV2,
    LandingCellMassV2,
    sample_ballistic_arc,
)


def test_ballistic_dataclasses_freeze_exact_public_fields() -> None:
    assert tuple(field.name for field in fields(BallisticStartV2)) == (
        "x_m", "y_m", "z_m"
    )
    assert tuple(field.name for field in fields(BallisticSampleV2)) == (
        "time_s", "x_m", "y_m", "z_m"
    )
    assert tuple(field.name for field in fields(LandingCellMassV2)) == (
        "cell", "probability_mass", "in_bounds"
    )
    start = BallisticStartV2(-0.0, 1.0, 2.0)
    assert start.x_m == 0.0
    assert not hasattr(start, "__dict__")
    with pytest.raises(FrozenInstanceError):
        start.x_m = 1.0
    with pytest.raises(TypeError, match="cell.*exact Cell"):
        LandingCellMassV2(object(), 0.5, True)
    with pytest.raises(TypeError, match="in_bounds.*exact bool"):
        LandingCellMassV2(Cell(0, 0), 0.5, 1)


def test_sample_ballistic_arc_has_exact_endpoints_and_lunar_range() -> None:
    start = BallisticStartV2(10.0, -2.0, 4.0)
    samples = sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 0.7)
    flight_time = 2.0 * ((3.0 * sin(pi / 4.0)) / 1.62)

    assert type(samples) is tuple
    assert samples[0] == BallisticSampleV2(0.0, 10.0, -2.0, 4.0)
    assert samples[-1].time_s == flight_time
    assert samples[-1].x_m == pytest.approx(10.0 + 9.0 / 1.62)
    assert samples[-1].y_m == pytest.approx(-2.0, abs=1.0e-15)
    assert samples[-1].z_m == start.z_m
    assert all(left.time_s < right.time_s for left, right in zip(samples, samples[1:]))
    assert all(
        right.time_s - left.time_s <= nextafter(0.7, float("inf"))
        for left, right in zip(samples, samples[1:])
    )


def test_sample_ballistic_arc_shortens_only_final_interval() -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    samples = sample_ballistic_arc(start, 1.5, pi / 6.0, 0.0, 1.62, 0.4)
    interior = tuple(sample.time_s for sample in samples[:-1])
    assert interior == tuple(index * 0.4 for index in range(len(interior)))
    assert samples[-1].time_s - samples[-2].time_s <= nextafter(0.4, float("inf"))

    two_samples = sample_ballistic_arc(start, 1.5, pi / 6.0, 0.0, 1.62, 99.0)
    assert len(two_samples) == 2
    assert two_samples[0].time_s == 0.0
    assert two_samples[-1].z_m == 0.0


@pytest.mark.parametrize(
    ("azimuth", "expected_signs"),
    [
        (0.0, (1, 0)),
        (pi / 2.0, (0, 1)),
        (pi, (-1, 0)),
        (3.0 * pi / 2.0, (0, -1)),
    ],
)
def test_sample_ballistic_arc_rotates_cardinal_azimuths(
    azimuth: float,
    expected_signs: tuple[int, int],
) -> None:
    end = sample_ballistic_arc(
        BallisticStartV2(0.0, 0.0, 1.0), 2.0, pi / 4.0, azimuth, 1.62, 0.25
    )[-1]
    dx, dy = end.x_m, end.y_m
    if expected_signs[0] == 0:
        assert dx == pytest.approx(0.0, abs=1.0e-15)
    else:
        assert dx * expected_signs[0] > 0.0
    if expected_signs[1] == 0:
        assert dy == pytest.approx(0.0, abs=1.0e-15)
    else:
        assert dy * expected_signs[1] > 0.0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("speed_mps", True, TypeError),
        ("speed_mps", 0.0, ValueError),
        ("speed_mps", float("nan"), ValueError),
        ("g_mps2", 0.0, ValueError),
        ("g_mps2", float("inf"), ValueError),
        ("dt_s", -1.0, ValueError),
        ("elevation_rad", 0.0, ValueError),
        ("elevation_rad", pi / 2.0, ValueError),
        ("azimuth_rad", float("-inf"), ValueError),
    ],
)
def test_sample_ballistic_arc_rejects_invalid_inputs(field, value, error) -> None:
    values = {
        "speed_mps": 3.0,
        "elevation_rad": pi / 4.0,
        "azimuth_rad": 0.0,
        "g_mps2": 1.62,
        "dt_s": 0.25,
    }
    values[field] = value
    with pytest.raises(error, match=field):
        sample_ballistic_arc(BallisticStartV2(0.0, 0.0, 0.0), **values)


def test_sample_ballistic_arc_checks_derived_limits_before_allocation() -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    assert MAX_BALLISTIC_SAMPLES_V2 == 100_000
    with pytest.raises(ValueError, match="sample_count"):
        sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 1.0e-12)
    with pytest.raises(ValueError, match="finite"):
        sample_ballistic_arc(start, 1.0e308, pi / 4.0, 0.0, 1.0e-308, 1.0)


def test_sample_ballistic_arc_reaudits_forged_exact_start() -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    object.__setattr__(start, "z_m", float("nan"))
    with pytest.raises(ValueError, match="z_m.*finite"):
        sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 0.25)
