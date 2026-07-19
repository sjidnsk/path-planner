from dataclasses import FrozenInstanceError, fields
from math import fsum, nextafter, pi, sin, ulp

import pytest

import path_planner.v2.ballistics as ballistics_module
from path_planner.core.models import Cell, WorldPoint
from path_planner.v2.ballistics import (
    MAX_BALLISTIC_SAMPLES_V2,
    MAX_LANDING_ZONE_CANDIDATES_V2,
    BallisticSampleV2,
    BallisticStartV2,
    LandingCellMassV2,
    landing_zone_cells,
    normal_interval_mass,
    sample_ballistic_arc,
)
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineGridGeometryV2


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


def test_sample_ballistic_arc_never_exceeds_dt_at_a_float_boundary() -> None:
    dt_s = 0.1
    speed_mps = 1.1 * 1.62 / (2.0 * sin(pi / 4.0))
    samples = sample_ballistic_arc(
        BallisticStartV2(0.0, 0.0, 0.0),
        speed_mps,
        pi / 4.0,
        0.0,
        1.62,
        dt_s,
    )
    assert samples[-1].time_s == 1.1
    assert len(samples) == 13

    times = tuple(sample.time_s for sample in samples)
    assert times[:3] == (0.0, dt_s, 2.0 * dt_s)
    for index, time_s in enumerate(times[1:-1], start=1):
        nominal_time = index * dt_s
        # Four local ULPs permit only representational correction, not a
        # global repartition of the cadence across all intervals.
        assert abs(time_s - nominal_time) <= 4.0 * ulp(nominal_time)

    gaps = tuple(right - left for left, right in zip(times, times[1:]))
    assert all(gap <= dt_s for gap in gaps)
    assert all(abs(gap - dt_s) <= 16.0 * ulp(dt_s) for gap in gaps[:-1])
    assert gaps[-1] < 0.5 * dt_s


@pytest.mark.parametrize(
    ("speed_mps", "elevation_rad", "g_mps2", "reason"),
    [
        (nextafter(0.0, 1.0), pi / 6.0, 1.62, "vertical speed"),
        (1.0e-200, pi / 4.0, 1.0e200, "vertical speed / gravity"),
    ],
)
def test_sample_ballistic_arc_rejects_positive_derived_underflow(
    speed_mps: float,
    elevation_rad: float,
    g_mps2: float,
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{reason}.*representable"):
        sample_ballistic_arc(
            BallisticStartV2(0.0, 0.0, 0.0),
            speed_mps,
            elevation_rad,
            0.0,
            g_mps2,
            1.0,
        )


@pytest.mark.parametrize(
    "start",
    [
        BallisticStartV2(1.0e308, 0.0, 0.0),
        BallisticStartV2(0.0, 0.0, 1.0e308),
    ],
)
def test_sample_ballistic_arc_rejects_absorbed_nonzero_endpoint_offset(
    start: BallisticStartV2,
) -> None:
    with pytest.raises(ValueError, match="representability"):
        sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 0.25)


def test_sample_ballistic_arc_rejects_absorbed_nonzero_interior_offset() -> None:
    with pytest.raises(ValueError, match="sample x_m.*representability"):
        sample_ballistic_arc(
            BallisticStartV2(1.0e16, 0.0, 0.0),
            3.0,
            pi / 4.0,
            0.0,
            1.62,
            0.1,
        )


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


@pytest.mark.parametrize("field", ("x_m", "y_m", "z_m"))
def test_sample_ballistic_arc_rejects_forged_start_with_deleted_field(
    field: str,
) -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    object.__delattr__(start, field)
    with pytest.raises(TypeError, match="start.*field"):
        sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 0.25)


@pytest.mark.parametrize("field", ("x", "y"))
def test_landing_cell_mass_rejects_forged_cell_with_deleted_field(
    field: str,
) -> None:
    cell = Cell(0, 0)
    object.__delattr__(cell, field)
    with pytest.raises(TypeError, match="cell.*field"):
        LandingCellMassV2(cell, 0.5, True)


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (-1.0, 1.0, 0.6826894921370859),
        (8.0, 9.0, 6.219831985865866e-16),
        (9.0, 10.0, 1.1285122074236006e-19),
    ],
)
def test_normal_interval_mass_is_stable_in_center_and_far_tail(
    lo: float,
    hi: float,
    expected: float,
) -> None:
    assert normal_interval_mass(lo, hi, 0.0, 1.0) == pytest.approx(
        expected, rel=1.0e-14, abs=0.0
    )


def test_normal_interval_mass_zero_width_symmetry_and_translation() -> None:
    assert normal_interval_mass(2.0, 2.0, 1.0, 0.5) == 0.0
    left = normal_interval_mass(-1.5, -0.5, -1.0, 0.25)
    right = normal_interval_mass(0.5, 1.5, 1.0, 0.25)
    translated = normal_interval_mass(8.5, 9.5, 9.0, 0.25)
    assert left == right == translated


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ((1.0, 0.0, 0.0, 1.0), ValueError, "lo.*hi"),
        ((0.0, 1.0, 0.0, 0.0), ValueError, "sigma"),
        ((0.0, 1.0, 0.0, True), TypeError, "sigma"),
        ((0.0, float("inf"), 0.0, 1.0), ValueError, "hi.*finite"),
    ],
)
def test_normal_interval_mass_rejects_invalid_contracts(values, error, message) -> None:
    with pytest.raises(error, match=message):
        normal_interval_mass(*values)


def _landing_key(item: LandingCellMassV2, geometry: FineGridGeometryV2, mean: WorldPoint):
    center_x = geometry.origin[0] + (item.cell.x + 0.5) * geometry.resolution_m
    center_y = geometry.origin[1] + (item.cell.y + 0.5) * geometry.resolution_m
    distance_sq = (center_x - mean.x) ** 2 + (center_y - mean.y) ** 2
    return (-item.probability_mass, distance_sq, item.cell.y, item.cell.x)


def test_landing_zone_is_shortest_global_raw_mass_prefix() -> None:
    geometry = FineGridGeometryV2(5, 5, origin=(-1.25, -1.25))
    mean = WorldPoint(0.0, 0.0)
    threshold = 0.99
    zone = landing_zone_cells(mean, 0.2, threshold, geometry)

    assert type(zone) is tuple
    assert zone == tuple(sorted(zone, key=lambda item: _landing_key(item, geometry, mean)))
    masses = tuple(item.probability_mass for item in zone)
    assert fsum(masses) >= threshold
    assert fsum(masses[:-1]) < threshold
    for item in zone:
        x_lo = geometry.origin[0] + item.cell.x * geometry.resolution_m
        y_lo = geometry.origin[1] + item.cell.y * geometry.resolution_m
        expected = normal_interval_mass(x_lo, x_lo + 0.5, mean.x, 0.2) * normal_interval_mass(
            y_lo, y_lo + 0.5, mean.y, 0.2
        )
        assert item.probability_mass == expected
        assert item.in_bounds is geometry.in_bounds(item.cell)


def test_landing_zone_boundary_ties_use_distance_y_x_order() -> None:
    geometry = FineGridGeometryV2(4, 4)
    zone = landing_zone_cells(WorldPoint(1.0, 1.0), 0.05, 0.99, geometry)
    assert tuple(item.cell for item in zone) == (
        Cell(1, 1),
        Cell(2, 1),
        Cell(1, 2),
        Cell(2, 2),
    )


def test_landing_zone_retains_oob_mass_without_renormalizing() -> None:
    geometry = FineGridGeometryV2(1, 1)
    zone = landing_zone_cells(WorldPoint(0.5, 0.5), 0.4, 0.99, geometry)
    in_bounds_mass = fsum(item.probability_mass for item in zone if item.in_bounds)
    assert any(item.in_bounds is False for item in zone)
    assert in_bounds_mass < 0.99
    assert fsum(item.probability_mass for item in zone) >= 0.99


def test_landing_zone_is_byte_stable_and_recomputes_disclosure_flags() -> None:
    geometry = FineGridGeometryV2(3, 3, origin=(-0.5, -0.5))
    first = landing_zone_cells(WorldPoint(0.25, 0.25), 0.17, 0.99, geometry)
    second = landing_zone_cells(WorldPoint(0.25, 0.25), 0.17, 0.99, geometry)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert tuple(item.in_bounds for item in first) == tuple(
        geometry.in_bounds(item.cell) for item in first
    )


def test_landing_zone_handles_legal_extremes_or_hits_public_cap(monkeypatch) -> None:
    geometry = FineGridGeometryV2(2, 2)
    one = landing_zone_cells(
        WorldPoint(0.25, 0.25),
        1.0e-6,
        nextafter(1.0, 0.0),
        geometry,
    )
    assert one == (LandingCellMassV2(Cell(0, 0), 1.0, True),)

    assert MAX_LANDING_ZONE_CANDIDATES_V2 == 1_000_000
    monkeypatch.setattr(ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", 9)
    with pytest.raises(ValueError, match="candidate.*9"):
        landing_zone_cells(WorldPoint(0.25, 0.25), 5.0, 0.99, geometry)


def test_landing_zone_delays_global_prefix_work_until_sparse_checkpoints(
    monkeypatch,
) -> None:
    prefix_scan_sizes: list[int] = []
    original = ballistics_module._shortest_prefix_length

    def tracking_prefix_length(
        masses: tuple[float, ...], threshold: float
    ) -> int | None:
        prefix_scan_sizes.append(len(masses))
        return original(masses, threshold)

    monkeypatch.setattr(ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", 225)
    monkeypatch.setattr(
        ballistics_module, "_shortest_prefix_length", tracking_prefix_length
    )
    with pytest.raises(ValueError, match="candidate.*225"):
        landing_zone_cells(
            WorldPoint(0.25, 0.25),
            50.0,
            0.99,
            FineGridGeometryV2(2, 2),
        )

    assert prefix_scan_sizes == [1, 9, 49, 225]


def test_landing_zone_visits_each_generated_perimeter_cell_once(
    monkeypatch,
) -> None:
    generated_cells: list[tuple[int, int]] = []
    original = ballistics_module._iter_square_perimeter_cells

    def tracking_perimeter(
        center_x: int, center_y: int, radius: int
    ):
        for coordinates in original(center_x, center_y, radius):
            generated_cells.append(coordinates)
            yield coordinates

    monkeypatch.setattr(ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", 49)
    monkeypatch.setattr(
        ballistics_module, "_iter_square_perimeter_cells", tracking_perimeter
    )
    with pytest.raises(ValueError, match="candidate.*49"):
        landing_zone_cells(
            WorldPoint(0.25, 0.25),
            50.0,
            0.99,
            FineGridGeometryV2(2, 2),
        )

    assert len(generated_cells) == 49
    assert len(set(generated_cells)) == 49
    assert set(generated_cells) == {
        (x_index, y_index)
        for y_index in range(-3, 4)
        for x_index in range(-3, 4)
    }


def test_landing_zone_reaudits_forged_exact_outer_objects() -> None:
    mean = WorldPoint(0.25, 0.25)
    object.__setattr__(mean, "x", float("nan"))
    with pytest.raises(ValueError, match="mean_xy.x.*finite"):
        landing_zone_cells(mean, 0.1, 0.99, FineGridGeometryV2(2, 2))

    geometry = FineGridGeometryV2(2, 2)
    object.__setattr__(geometry, "width", True)
    with pytest.raises(TypeError, match="geometry.width.*exact int"):
        landing_zone_cells(WorldPoint(0.25, 0.25), 0.1, 0.99, geometry)


@pytest.mark.parametrize("field", ("x", "y"))
def test_landing_zone_rejects_forged_mean_with_deleted_field(field: str) -> None:
    mean = WorldPoint(0.25, 0.25)
    object.__delattr__(mean, field)
    with pytest.raises(TypeError, match="mean_xy.*field"):
        landing_zone_cells(mean, 0.1, 0.99, FineGridGeometryV2(2, 2))


@pytest.mark.parametrize(
    "field", ("width", "height", "origin", "frame_id", "resolution_m")
)
def test_landing_zone_rejects_forged_geometry_with_deleted_field(field: str) -> None:
    geometry = FineGridGeometryV2(2, 2)
    object.__delattr__(geometry, field)
    with pytest.raises(TypeError, match="geometry.*field"):
        landing_zone_cells(WorldPoint(0.25, 0.25), 0.1, 0.99, geometry)


@pytest.mark.parametrize(
    ("sigma", "threshold", "error"),
    [
        (0.0, 0.99, ValueError),
        (True, 0.99, TypeError),
        (0.1, 0.0, ValueError),
        (0.1, 1.0, ValueError),
        (0.1, float("nan"), ValueError),
    ],
)
def test_landing_zone_rejects_invalid_probability_inputs(sigma, threshold, error) -> None:
    with pytest.raises(error):
        landing_zone_cells(WorldPoint(0.25, 0.25), sigma, threshold, FineGridGeometryV2(2, 2))
