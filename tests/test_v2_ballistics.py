from dataclasses import FrozenInstanceError, fields
from inspect import Parameter, signature
from math import acos, asin, cos, fsum, nextafter, pi, sin, ulp

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


def test_sample_ballistic_arc_keeps_independent_anchors_within_four_ulps() -> None:
    flight_time = 1.675
    dt_s = 0.1
    speed_mps = flight_time * 1.62 / (2.0 * sin(pi / 4.0))
    samples = sample_ballistic_arc(
        BallisticStartV2(0.0, 0.0, 0.0),
        speed_mps,
        pi / 4.0,
        0.0,
        1.62,
        dt_s,
    )

    times = tuple(sample.time_s for sample in samples)
    assert times[-1] == flight_time
    assert all(left < right for left, right in zip(times, times[1:]))
    assert all(right - left <= dt_s for left, right in zip(times, times[1:]))
    for index in range(1, 17):
        nominal_anchor = index * dt_s
        assert any(
            abs(time_s - nominal_anchor) <= 4.0 * ulp(nominal_anchor)
            for time_s in times[1:-1]
        )


def test_sample_ballistic_arc_inserts_bounded_repairs_for_valid_schedule() -> None:
    flight_time = 1.7
    dt_s = 0.1
    speed_mps = flight_time * 1.62 / (2.0 * sin(pi / 4.0))
    samples = sample_ballistic_arc(
        BallisticStartV2(0.0, 0.0, 0.0),
        speed_mps,
        pi / 4.0,
        0.0,
        1.62,
        dt_s,
    )

    times = tuple(sample.time_s for sample in samples)
    assert times[-1] == flight_time
    assert len(times) > 18
    assert all(left < right for left, right in zip(times, times[1:]))
    assert all(right - left <= dt_s for left, right in zip(times, times[1:]))
    for index in range(1, 17):
        nominal_anchor = index * dt_s
        assert any(
            abs(time_s - nominal_anchor) <= 4.0 * ulp(nominal_anchor)
            for time_s in times[1:-1]
        )


def test_sample_ballistic_arc_bridges_endpoint_anchor_collision() -> None:
    dt_s = 0.1
    flight_time = dt_s * 24
    speed_mps = flight_time * 1.62 / (2.0 * sin(pi / 4.0))
    exact_interval_lower_bound = ballistics_module._exact_positive_ratio_ceil(
        flight_time, dt_s
    )
    assert flight_time == 2.4000000000000004
    assert exact_interval_lower_bound == 25

    samples = sample_ballistic_arc(
        BallisticStartV2(0.0, 0.0, 0.0),
        speed_mps,
        pi / 4.0,
        0.0,
        1.62,
        dt_s,
    )

    times = tuple(sample.time_s for sample in samples)
    assert exact_interval_lower_bound + 1 <= len(times) <= MAX_BALLISTIC_SAMPLES_V2
    assert times[0] == 0.0
    assert times[-1] == flight_time
    assert all(left < right for left, right in zip(times, times[1:]))
    assert all(right - left <= dt_s for left, right in zip(times, times[1:]))
    for index in range(1, exact_interval_lower_bound):
        nominal_anchor = index * dt_s
        assert any(
            abs(time_s - nominal_anchor) <= 4.0 * ulp(nominal_anchor)
            for time_s in times[1:]
        )


def test_sample_ballistic_arc_counts_repairs_against_sample_cap(monkeypatch) -> None:
    flight_time = 1.7
    speed_mps = flight_time * 1.62 / (2.0 * sin(pi / 4.0))
    monkeypatch.setattr(ballistics_module, "MAX_BALLISTIC_SAMPLES_V2", 18)

    with pytest.raises(ValueError, match="sample_count.*18"):
        sample_ballistic_arc(
            BallisticStartV2(0.0, 0.0, 0.0),
            speed_mps,
            pi / 4.0,
            0.0,
            1.62,
            0.1,
        )


@pytest.mark.parametrize(
    ("axis", "azimuth_rad"),
    [
        ("x", acos(0.25)),
        ("y", asin(0.25)),
    ],
)
def test_sample_ballistic_arc_rejects_direction_product_underflow(
    axis: str,
    azimuth_rad: float,
) -> None:
    minimum_subnormal = nextafter(0.0, 1.0)
    speed_mps = 2.0 * minimum_subnormal
    horizontal_speed = speed_mps * cos(pi / 3.0)
    direction_factor = cos(azimuth_rad) if axis == "x" else sin(azimuth_rad)
    assert horizontal_speed == minimum_subnormal
    assert direction_factor == 0.25
    assert horizontal_speed * direction_factor == 0.0

    with pytest.raises(ValueError, match=rf"{axis} velocity.*representability"):
        sample_ballistic_arc(
            BallisticStartV2(0.0, 0.0, 0.0),
            speed_mps,
            pi / 3.0,
            azimuth_rad,
            2.0 * minimum_subnormal,
            2.0,
        )


@pytest.mark.parametrize(
    ("axis", "azimuth_rad"),
    [
        ("x", pi / 3.0),
        ("y", pi / 6.0),
    ],
)
def test_sample_ballistic_arc_rejects_displacement_product_underflow(
    axis: str,
    azimuth_rad: float,
) -> None:
    minimum_subnormal = nextafter(0.0, 1.0)
    speed_mps = 6.0 * minimum_subnormal
    elevation_rad = acos(0.25)
    horizontal_speed = speed_mps * cos(elevation_rad)
    velocity_component = horizontal_speed * (
        cos(azimuth_rad) if axis == "x" else sin(azimuth_rad)
    )
    vertical_speed = speed_mps * sin(elevation_rad)
    gravity = 24.0 * minimum_subnormal
    flight_time = 2.0 * (vertical_speed / gravity)
    assert velocity_component == minimum_subnormal
    assert flight_time == 0.5
    assert velocity_component * flight_time == 0.0

    with pytest.raises(
        ValueError, match=rf"landing {axis} displacement.*representability"
    ):
        sample_ballistic_arc(
            BallisticStartV2(0.0, 0.0, 0.0),
            speed_mps,
            elevation_rad,
            azimuth_rad,
            gravity,
            0.5,
        )


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


def test_landing_zone_uses_frozen_cell_center_order_for_equal_mass_ties() -> None:
    geometry = FineGridGeometryV2(2, 2, origin=(-0.2, 0.0))
    mean = WorldPoint(-0.95, 0.25)

    result = landing_zone_cells(mean, 0.3, 0.4, geometry)

    assert tuple(item.cell for item in result) == (Cell(-2, 0), Cell(-1, 0))
    public_keys = tuple(
        (
            -item.probability_mass,
            (
                geometry.origin[0]
                + (item.cell.x + 0.5) * geometry.resolution_m
                - mean.x
            )
            ** 2
            + (
                geometry.origin[1]
                + (item.cell.y + 0.5) * geometry.resolution_m
                - mean.y
            )
            ** 2,
            item.cell.y,
            item.cell.x,
        )
        for item in result
    )
    assert public_keys == tuple(sorted(public_keys))


def test_landing_zone_rejects_absorbed_cell_center_offset() -> None:
    origin = float(2**51)
    geometry = FineGridGeometryV2(2, 2, origin=(origin, origin))

    with pytest.raises(ValueError, match="cell center.*representable"):
        landing_zone_cells(WorldPoint(origin, origin), 0.1, 0.4, geometry)


@pytest.mark.parametrize(
    ("origin", "mean"),
    [
        ((0.5, -0.5), WorldPoint(0.75, -0.25)),
        ((-1.0, 1.5), WorldPoint(-0.25, 2.25)),
        ((1.5, 2.0), WorldPoint(2.75, 3.25)),
    ],
)
def test_landing_zone_public_keys_are_reconstructable_for_nonzero_origins(
    origin: tuple[float, float],
    mean: WorldPoint,
) -> None:
    geometry = FineGridGeometryV2(3, 3, origin=origin)

    result = landing_zone_cells(mean, 0.3, 0.9, geometry)

    public_keys = []
    for item in result:
        center_x = origin[0] + (item.cell.x + 0.5) * geometry.resolution_m
        center_y = origin[1] + (item.cell.y + 0.5) * geometry.resolution_m
        public_keys.append(
            (
                -item.probability_mass,
                (center_x - mean.x) ** 2 + (center_y - mean.y) ** 2,
                item.cell.y,
                item.cell.x,
            )
        )
    assert tuple(public_keys) == tuple(sorted(public_keys))


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


class _CapIntSubclass(int):
    pass


class _CapCoercible:
    def __int__(self) -> int:
        return 8


def _ballistic_call_args() -> tuple[object, ...]:
    return (
        BallisticStartV2(0.0, 0.0, 0.5),
        3.0,
        pi / 4.0,
        0.0,
        1.62,
        0.25,
    )


def _landing_call_args() -> tuple[object, ...]:
    return (
        WorldPoint(0.25, 0.25),
        0.05,
        0.99,
        FineGridGeometryV2(4, 4),
    )


def test_capped_ballistic_helper_freezes_signature_output_and_legacy_delegation(
    monkeypatch,
) -> None:
    capped = getattr(ballistics_module, "sample_ballistic_arc_capped_v2")
    legacy_parameters = tuple(signature(sample_ballistic_arc).parameters.values())
    capped_parameters = tuple(signature(capped).parameters.values())
    assert tuple(parameter.name for parameter in legacy_parameters) == (
        "start",
        "speed_mps",
        "elevation_rad",
        "azimuth_rad",
        "g_mps2",
        "dt_s",
    )
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is Parameter.empty
        for parameter in legacy_parameters
    )
    assert tuple(parameter.name for parameter in capped_parameters) == (
        "start",
        "speed_mps",
        "elevation_rad",
        "azimuth_rad",
        "g_mps2",
        "dt_s",
        "max_sample_count",
    )
    assert capped_parameters[:-1] == legacy_parameters
    assert capped_parameters[-1].kind is Parameter.KEYWORD_ONLY
    assert capped_parameters[-1].default is Parameter.empty
    assert signature(capped).return_annotation == signature(
        sample_ballistic_arc
    ).return_annotation

    args = _ballistic_call_args()
    explicit = capped(*args, max_sample_count=MAX_BALLISTIC_SAMPLES_V2)
    assert canonical_json_bytes(sample_ballistic_arc(*args)) == canonical_json_bytes(
        explicit
    )

    sentinel = (BallisticSampleV2(0.0, 1.0, 2.0, 3.0),)
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_capped(*call_args, **call_kwargs):
        observed.append((call_args, call_kwargs))
        return sentinel

    monkeypatch.setattr(
        ballistics_module, "sample_ballistic_arc_capped_v2", fake_capped
    )
    assert sample_ballistic_arc(*args) is sentinel
    assert observed == [(args, {"max_sample_count": MAX_BALLISTIC_SAMPLES_V2})]


def test_capped_landing_helper_freezes_signature_output_and_legacy_delegation(
    monkeypatch,
) -> None:
    capped = getattr(ballistics_module, "landing_zone_cells_capped_v2")
    legacy_parameters = tuple(signature(landing_zone_cells).parameters.values())
    capped_parameters = tuple(signature(capped).parameters.values())
    assert tuple(parameter.name for parameter in legacy_parameters) == (
        "mean_xy",
        "sigma_m",
        "probability_threshold",
        "geometry",
    )
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is Parameter.empty
        for parameter in legacy_parameters
    )
    assert tuple(parameter.name for parameter in capped_parameters) == (
        "mean_xy",
        "sigma_m",
        "probability_threshold",
        "geometry",
        "max_candidate_count",
    )
    assert capped_parameters[:-1] == legacy_parameters
    assert capped_parameters[-1].kind is Parameter.KEYWORD_ONLY
    assert capped_parameters[-1].default is Parameter.empty
    assert signature(capped).return_annotation == signature(
        landing_zone_cells
    ).return_annotation

    args = _landing_call_args()
    explicit = capped(*args, max_candidate_count=MAX_LANDING_ZONE_CANDIDATES_V2)
    assert canonical_json_bytes(landing_zone_cells(*args)) == canonical_json_bytes(
        explicit
    )

    sentinel = (LandingCellMassV2(Cell(7, 8), 1.0, False),)
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_capped(*call_args, **call_kwargs):
        observed.append((call_args, call_kwargs))
        return sentinel

    monkeypatch.setattr(ballistics_module, "landing_zone_cells_capped_v2", fake_capped)
    assert landing_zone_cells(*args) is sentinel
    assert observed == [
        (args, {"max_candidate_count": MAX_LANDING_ZONE_CANDIDATES_V2})
    ]


def test_capped_ballistic_helper_rejects_non_exact_caps_before_materialization(
    monkeypatch,
) -> None:
    capped = getattr(ballistics_module, "sample_ballistic_arc_capped_v2")
    materialization_calls = 0

    def forbidden_materialization(*_args, **_kwargs):
        nonlocal materialization_calls
        materialization_calls += 1
        raise AssertionError("interior sample materialization must not run")

    monkeypatch.setattr(
        ballistics_module, "_interior_sample_times", forbidden_materialization
    )
    for value in (True, _CapIntSubclass(8), _CapCoercible()):
        with pytest.raises(TypeError, match="max_sample_count.*exact int"):
            capped(*_ballistic_call_args(), max_sample_count=value)
    for value in (0, -1):
        with pytest.raises(ValueError, match="max_sample_count.*positive"):
            capped(*_ballistic_call_args(), max_sample_count=value)
    assert materialization_calls == 0


def test_capped_landing_helper_rejects_non_exact_caps_before_materialization(
    monkeypatch,
) -> None:
    capped = getattr(ballistics_module, "landing_zone_cells_capped_v2")
    materialization_calls = 0

    def forbidden_materialization(*_args, **_kwargs):
        nonlocal materialization_calls
        materialization_calls += 1
        raise AssertionError("landing candidate materialization must not run")

    monkeypatch.setattr(
        ballistics_module,
        "_iter_square_perimeter_cells",
        forbidden_materialization,
    )
    for value in (True, _CapIntSubclass(8), _CapCoercible()):
        with pytest.raises(TypeError, match="max_candidate_count.*exact int"):
            capped(*_landing_call_args(), max_candidate_count=value)
    for value in (0, -1):
        with pytest.raises(ValueError, match="max_candidate_count.*positive"):
            capped(*_landing_call_args(), max_candidate_count=value)
    assert materialization_calls == 0


def test_capped_helpers_use_only_explicit_caps_not_mutable_legacy_globals(
    monkeypatch,
) -> None:
    ballistic_capped = getattr(ballistics_module, "sample_ballistic_arc_capped_v2")
    landing_capped = getattr(ballistics_module, "landing_zone_cells_capped_v2")
    ballistic_args = _ballistic_call_args()
    landing_args = _landing_call_args()
    expected_ballistic = ballistic_capped(
        *ballistic_args, max_sample_count=MAX_BALLISTIC_SAMPLES_V2
    )
    expected_landing = landing_capped(
        *landing_args, max_candidate_count=MAX_LANDING_ZONE_CANDIDATES_V2
    )

    monkeypatch.setattr(ballistics_module, "MAX_BALLISTIC_SAMPLES_V2", True)
    monkeypatch.setattr(ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", True)
    assert canonical_json_bytes(
        ballistic_capped(
            *ballistic_args, max_sample_count=MAX_BALLISTIC_SAMPLES_V2
        )
    ) == canonical_json_bytes(expected_ballistic)
    assert canonical_json_bytes(
        landing_capped(
            *landing_args, max_candidate_count=MAX_LANDING_ZONE_CANDIDATES_V2
        )
    ) == canonical_json_bytes(expected_landing)

    monkeypatch.setattr(ballistics_module, "MAX_BALLISTIC_SAMPLES_V2", 10**9)
    monkeypatch.setattr(
        ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", 10**9
    )
    with pytest.raises(ValueError, match="sample_count.*2"):
        ballistic_capped(*ballistic_args, max_sample_count=2)
    with pytest.raises(ValueError, match="candidate.*1"):
        landing_capped(
            WorldPoint(0.25, 0.25),
            5.0,
            0.99,
            FineGridGeometryV2(2, 2),
            max_candidate_count=1,
        )


def test_capped_ballistic_helper_checks_valid_small_cap_before_materialization(
    monkeypatch,
) -> None:
    capped = getattr(ballistics_module, "sample_ballistic_arc_capped_v2")
    materialization_calls = 0

    def forbidden_materialization(*_args, **_kwargs):
        nonlocal materialization_calls
        materialization_calls += 1
        raise AssertionError("interior sample materialization must not run")

    monkeypatch.setattr(
        ballistics_module, "_interior_sample_times", forbidden_materialization
    )
    with pytest.raises(ValueError, match="sample_count.*2"):
        capped(*_ballistic_call_args(), max_sample_count=2)
    assert materialization_calls == 0


def test_capped_ballistic_helper_uses_explicit_cap_for_inner_repair_budget(
    monkeypatch,
) -> None:
    capped = getattr(ballistics_module, "sample_ballistic_arc_capped_v2")
    flight_time = 1.7
    speed_mps = flight_time * 1.62 / (2.0 * sin(pi / 4.0))
    args = (
        BallisticStartV2(0.0, 0.0, 0.0),
        speed_mps,
        pi / 4.0,
        0.0,
        1.62,
        0.1,
    )
    monkeypatch.setattr(ballistics_module, "MAX_BALLISTIC_SAMPLES_V2", 10**9)

    with pytest.raises(ValueError, match="sample_count.*18"):
        capped(*args, max_sample_count=18)
    samples = capped(*args, max_sample_count=19)
    assert len(samples) == 19
    assert samples[0].time_s == 0.0
    assert samples[-1].time_s == flight_time


def test_legacy_wrappers_keep_input_audit_precedence_over_invalid_global_caps(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ballistics_module, "MAX_BALLISTIC_SAMPLES_V2", True)
    with pytest.raises(TypeError, match="start.*exact BallisticStartV2"):
        sample_ballistic_arc(object(), 3.0, pi / 4.0, 0.0, 1.62, 0.25)

    monkeypatch.setattr(ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", True)
    with pytest.raises(TypeError, match="mean_xy.*exact WorldPoint"):
        landing_zone_cells(object(), 0.05, 0.99, FineGridGeometryV2(4, 4))
