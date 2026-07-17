from math import ceil, copysign, cos, floor, hypot, isfinite, nextafter, pi, sin, sqrt

import pytest

import path_planner.v2.geometry as geometry_module
from path_planner.core import Cell, WorldPoint
from path_planner.search import (
    MotionPrimitive,
    Pose2D,
    PoseTransition,
    replay_motion_primitive,
)
from path_planner.v2.geometry import (
    conservative_wheel_pose_cells,
    conservative_wheel_sweep_cells,
    convex_hull_xy,
    dense_wheel_replay_step_count,
    oriented_rectangle_cells,
    point_margin_to_convex_polygon,
    sample_pose_sweep,
)
from path_planner.v2.contracts import PoseStateV2
from path_planner.v2.terrain import FineGridGeometryV2


def test_shared_geometry_public_surface_exists() -> None:
    assert hasattr(geometry_module, "convex_hull_xy")
    assert hasattr(geometry_module, "point_margin_to_convex_polygon")
    assert hasattr(geometry_module, "oriented_rectangle_cells")
    assert hasattr(geometry_module, "sample_pose_sweep")


def test_dense_replay_count_uses_body_point_motion_bound() -> None:
    primitive = MotionPrimitive("curve", 0.5, pi / 4.0, 2.0)
    radius = hypot(0.612 / 2.0, 0.580 / 2.0)

    steps = dense_wheel_replay_step_count(
        primitive,
        body_length_m=0.612,
        body_width_m=0.580,
        safety_margin_m=0.0,
        resolution_m=0.5,
    )

    assert steps == ceil((abs(0.5) + radius * abs(pi / 4.0)) * 2.0 / (0.5 / 8.0))


def test_pose_contact_cells_are_sorted_unique_and_not_clamped() -> None:
    geometry = FineGridGeometryV2(
        width=4,
        height=3,
        origin=(10.0, -2.0),
        frame_id="moon",
    )

    cells = conservative_wheel_pose_cells(
        Pose2D(10.05, -1.25, 0.0),
        geometry,
        body_length_m=0.612,
        body_width_m=0.580,
        safety_margin_m=0.0,
    )

    assert Cell(-1, 1) in cells
    assert cells == tuple(sorted(set(cells), key=lambda cell: (cell.y, cell.x)))


def test_dense_sweep_contains_intermediate_cells_beyond_endpoint_footprints() -> None:
    geometry = FineGridGeometryV2(width=10, height=5, frame_id="moon")
    start = Pose2D(1.25, 1.25, 0.0)
    primitive = MotionPrimitive("long_forward", 1.0, 0.0, 2.0)
    kwargs = {
        "body_length_m": 0.612,
        "body_width_m": 0.580,
        "safety_margin_m": 0.0,
    }

    swept = conservative_wheel_sweep_cells(start, primitive, geometry, **kwargs)
    start_cells = conservative_wheel_pose_cells(start, geometry, **kwargs)
    end_cells = conservative_wheel_pose_cells(Pose2D(3.25, 1.25, 0.0), geometry, **kwargs)

    assert Cell(4, 2) in swept
    assert Cell(4, 2) not in start_cells
    assert Cell(4, 2) not in end_cells


def test_pose_contact_uses_closed_nextafter_boundary() -> None:
    geometry = FineGridGeometryV2(width=2, height=2, frame_id="moon")
    dimension = 2.0 * (0.5 - geometry.resolution_m / sqrt(2.0))

    cells = conservative_wheel_pose_cells(
        Pose2D(0.25, 0.25, 0.0),
        geometry,
        body_length_m=dimension,
        body_width_m=dimension,
        safety_margin_m=0.0,
    )

    assert Cell(-1, 0) in cells
    assert Cell(0, -1) in cells


def test_dense_replay_rejects_one_step_beyond_public_cap() -> None:
    speed_for_100001_steps = 100_001 * (0.5 / 8.0)
    primitive = MotionPrimitive("too_dense", speed_for_100001_steps, 0.0, 1.0)

    with pytest.raises(ValueError, match="100000|replay cap"):
        dense_wheel_replay_step_count(
            primitive,
            body_length_m=0.612,
            body_width_m=0.580,
            safety_margin_m=0.0,
            resolution_m=0.5,
        )


def test_dense_sweep_checks_deadline_during_public_replay() -> None:
    checks = iter((False, True))

    with pytest.raises(TimeoutError, match="deadline"):
        conservative_wheel_sweep_cells(
            Pose2D(1.25, 1.25, 0.0),
            MotionPrimitive("forward", 1.0, 0.0, 1.0),
            FineGridGeometryV2(width=6, height=6, frame_id="moon"),
            body_length_m=0.612,
            body_width_m=0.580,
            safety_margin_m=0.0,
            deadline_checker=lambda: next(checks),
        )


@pytest.mark.parametrize(
    "checks",
    [
        (False, False, True),
        (False, False, False, True),
    ],
)
def test_idle_sweep_checks_deadline_before_endpoint_and_return(checks) -> None:
    values = iter(checks)

    with pytest.raises(TimeoutError, match="deadline"):
        conservative_wheel_sweep_cells(
            Pose2D(1.25, 1.25, 0.0),
            MotionPrimitive("idle", 0.0, 0.0, 0.1),
            FineGridGeometryV2(width=6, height=6, frame_id="moon"),
            body_length_m=0.612,
            body_width_m=0.580,
            safety_margin_m=0.0,
            deadline_checker=lambda: next(values),
        )


def test_idle_sweep_endpoint_deadline_requires_exact_bool() -> None:
    values = iter((False, False, 1))

    with pytest.raises(TypeError, match="exact bool"):
        conservative_wheel_sweep_cells(
            Pose2D(1.25, 1.25, 0.0),
            MotionPrimitive("idle", 0.0, 0.0, 0.1),
            FineGridGeometryV2(width=6, height=6, frame_id="moon"),
            body_length_m=0.612,
            body_width_m=0.580,
            safety_margin_m=0.0,
            deadline_checker=lambda: next(values),
        )


def test_exact_public_replay_cap_remains_executable(monkeypatch) -> None:
    primitive = MotionPrimitive("at_cap", 62_500.0, 0.0, 0.1)
    observed = {}

    def replay_at_cap(start, control, integration_dt_s, *, deadline_checker=None):
        replay = replay_motion_primitive(
            start,
            control,
            integration_dt_s,
            deadline_checker=deadline_checker,
        )
        observed["integration_dt_s"] = integration_dt_s
        observed["steps"] = len(replay.samples) - 1
        return PoseTransition(
            start=start,
            primitive=control,
            samples=(start, start),
            end=start,
            distance_m=0.0,
            absolute_heading_change_rad=0.0,
        )

    monkeypatch.setattr(geometry_module, "replay_motion_primitive", replay_at_cap)

    cells = conservative_wheel_sweep_cells(
        Pose2D(1.25, 1.25, 0.0),
        primitive,
        FineGridGeometryV2(width=6, height=6, frame_id="moon"),
        body_length_m=0.612,
        body_width_m=0.580,
        safety_margin_m=0.0,
    )

    assert dense_wheel_replay_step_count(
        primitive,
        body_length_m=0.612,
        body_width_m=0.580,
        safety_margin_m=0.0,
        resolution_m=0.5,
    ) == 100_000
    assert observed["integration_dt_s"] > 0.1 / 100_000
    assert observed["steps"] == 100_000
    assert cells


def test_one_step_max_finite_interval_does_not_promote_replay_dt_to_infinity(
    monkeypatch,
) -> None:
    duration = float.fromhex("0x1.fffffffffffffp+1023")
    observed = {}

    def capture_replay(start, control, integration_dt_s, *, deadline_checker=None):
        observed["integration_dt_s"] = integration_dt_s
        return PoseTransition(
            start=start,
            primitive=control,
            samples=(start, start),
            end=start,
            distance_m=0.0,
            absolute_heading_change_rad=0.0,
        )

    monkeypatch.setattr(geometry_module, "replay_motion_primitive", capture_replay)

    conservative_wheel_sweep_cells(
        Pose2D(1.25, 1.25, 0.0),
        MotionPrimitive("max_duration_idle", 0.0, 0.0, duration),
        FineGridGeometryV2(width=6, height=6, frame_id="moon"),
        body_length_m=0.612,
        body_width_m=0.580,
        safety_margin_m=0.0,
    )

    assert isfinite(observed["integration_dt_s"])
    assert observed["integration_dt_s"] == duration


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"body_length_m": True}, TypeError, "body_length_m"),
        ({"body_width_m": 0.0}, ValueError, "positive"),
        ({"safety_margin_m": -0.1}, ValueError, "nonnegative"),
        ({"resolution_m": nextafter(0.0, 1.0)}, ValueError, "replay cap"),
    ],
)
def test_dense_replay_rejects_invalid_geometry_inputs(overrides, error, message) -> None:
    values = {
        "body_length_m": 0.612,
        "body_width_m": 0.580,
        "safety_margin_m": 0.0,
        "resolution_m": 0.5,
    }
    values.update(overrides)

    with pytest.raises(error, match=message):
        dense_wheel_replay_step_count(
            MotionPrimitive("forward", 1.0, 0.0, 1.0),
            **values,
        )


def test_dense_replay_converts_huge_real_overflow_to_stable_finite_error() -> None:
    with pytest.raises(ValueError, match="body_length_m.*finite"):
        dense_wheel_replay_step_count(
            MotionPrimitive("forward", 1.0, 0.0, 1.0),
            body_length_m=10**10_000,
            body_width_m=0.580,
            safety_margin_m=0.0,
            resolution_m=0.5,
        )


def test_pose_contact_rejects_nonfinite_normalized_bounds() -> None:
    geometry = FineGridGeometryV2(
        width=2,
        height=2,
        origin=(-1.0e308, -1.0e308),
        frame_id="moon",
    )

    with pytest.raises(ValueError, match="finite"):
        conservative_wheel_pose_cells(
            Pose2D(1.0e308, 1.0e308, 0.0),
            geometry,
            body_length_m=0.612,
            body_width_m=0.580,
            safety_margin_m=0.0,
        )


def test_pose_contact_rejects_candidate_enumeration_above_public_bound(
    monkeypatch,
) -> None:
    def forbidden_range(*_args):
        raise AssertionError("candidate range entered before bound check")

    monkeypatch.setattr(geometry_module, "range", forbidden_range, raising=False)

    with pytest.raises(ValueError, match="candidate.*100000"):
        conservative_wheel_pose_cells(
            Pose2D(1.25, 1.25, 0.0),
            FineGridGeometryV2(width=2, height=2, frame_id="moon"),
            body_length_m=1.0e9,
            body_width_m=0.580,
            safety_margin_m=0.0,
        )


@pytest.mark.parametrize(
    "primitive",
    [
        MotionPrimitive("straight", 0.8, 0.0, 0.75),
        MotionPrimitive("curve", 0.6, 0.8, 0.75),
    ],
)
def test_sweep_conservatively_contains_sampled_rotated_body_points(primitive) -> None:
    geometry = FineGridGeometryV2(
        width=12,
        height=12,
        origin=(10.0, -3.0),
        frame_id="moon",
    )
    start = Pose2D(12.25, -0.75, 0.37)
    length = 0.612
    width = 0.580
    margin = 0.05
    sweep = conservative_wheel_sweep_cells(
        start,
        primitive,
        geometry,
        body_length_m=length,
        body_width_m=width,
        safety_margin_m=margin,
    )
    dense_truth = replay_motion_primitive(start, primitive, 0.025)
    local_x_values = tuple((length / 2.0 + margin) * value for value in (-1.0, -0.5, 0.0, 0.5, 1.0))
    local_y_values = tuple((width / 2.0 + margin) * value for value in (-1.0, -0.5, 0.0, 0.5, 1.0))

    for pose in dense_truth.samples:
        cos_t = cos(pose.theta_rad)
        sin_t = sin(pose.theta_rad)
        for local_y in local_y_values:
            for local_x in local_x_values:
                world_x = pose.x_m + local_x * cos_t - local_y * sin_t
                world_y = pose.y_m + local_x * sin_t + local_y * cos_t
                cell = Cell(
                    floor((world_x - geometry.origin[0]) / geometry.resolution_m),
                    floor((world_y - geometry.origin[1]) / geometry.resolution_m),
                )
                assert cell in sweep

    assert sweep == conservative_wheel_sweep_cells(
        start,
        primitive,
        geometry,
        body_length_m=length,
        body_width_m=width,
        safety_margin_m=margin,
    )


def test_convex_hull_is_permutation_invariant_canonical_and_deduplicated() -> None:
    points = (
        WorldPoint(2.0, 2.0),
        WorldPoint(0.0, 0.0),
        WorldPoint(1.0, 1.0),
        WorldPoint(0.0, 2.0),
        WorldPoint(2.0, 0.0),
        WorldPoint(0.0, 0.0),
        WorldPoint(2.0, 2.0),
    )
    expected = (
        WorldPoint(0.0, 0.0),
        WorldPoint(2.0, 0.0),
        WorldPoint(2.0, 2.0),
        WorldPoint(0.0, 2.0),
    )

    assert convex_hull_xy(points) == expected
    assert convex_hull_xy(tuple(reversed(points))) == expected
    assert isinstance(convex_hull_xy(points), tuple)


def test_convex_hull_handles_empty_single_and_collinear_inputs() -> None:
    assert convex_hull_xy(()) == ()
    single = convex_hull_xy((WorldPoint(-0.0, -0.0), WorldPoint(0.0, 0.0)))
    assert single == (WorldPoint(0.0, 0.0),)
    assert copysign(1.0, single[0].x) == 1.0
    assert copysign(1.0, single[0].y) == 1.0
    assert convex_hull_xy(
        (
            WorldPoint(2.0, 2.0),
            WorldPoint(-1.0, -1.0),
            WorldPoint(0.0, 0.0),
            WorldPoint(1.0, 1.0),
        )
    ) == (WorldPoint(-1.0, -1.0), WorldPoint(2.0, 2.0))


def test_convex_hull_requires_exact_finite_world_points_and_finite_cross() -> None:
    class DerivedWorldPoint(WorldPoint):
        pass

    with pytest.raises(TypeError, match="exact WorldPoint"):
        convex_hull_xy((DerivedWorldPoint(0.0, 0.0),))
    with pytest.raises(TypeError, match="finite real"):
        convex_hull_xy((WorldPoint(True, 0.0),))
    with pytest.raises(ValueError, match="finite"):
        convex_hull_xy((WorldPoint(float("nan"), 0.0),))
    with pytest.raises(ValueError, match="finite"):
        convex_hull_xy(
            (
                WorldPoint(-1.0e308, 0.0),
                WorldPoint(1.0e308, 0.0),
                WorldPoint(0.0, 1.0),
            )
        )


def test_point_margin_uses_canonical_ccw_signed_edge_distance() -> None:
    unordered_closed_square = (
        WorldPoint(1.0, 1.0),
        WorldPoint(0.0, 0.0),
        WorldPoint(0.0, 1.0),
        WorldPoint(1.0, 0.0),
        WorldPoint(1.0, 1.0),
    )

    assert point_margin_to_convex_polygon(
        WorldPoint(0.5, 0.5), unordered_closed_square
    ) == 0.5
    edge_margin = point_margin_to_convex_polygon(
        WorldPoint(0.0, 0.5), unordered_closed_square
    )
    assert edge_margin == 0.0
    assert copysign(1.0, edge_margin) == 1.0
    assert point_margin_to_convex_polygon(
        WorldPoint(-0.25, 0.5), unordered_closed_square
    ) == -0.25


def test_point_margin_preserves_exact_support_threshold_sides() -> None:
    square = (
        WorldPoint(0.0, 0.0),
        WorldPoint(1.0, 0.0),
        WorldPoint(1.0, 1.0),
        WorldPoint(0.0, 1.0),
    )
    below = nextafter(0.05, 0.0)
    above = nextafter(0.05, 1.0)

    assert point_margin_to_convex_polygon(WorldPoint(below, 0.5), square) < 0.05
    assert point_margin_to_convex_polygon(WorldPoint(0.05, 0.5), square) == 0.05
    assert point_margin_to_convex_polygon(WorldPoint(above, 0.5), square) > 0.05


@pytest.mark.parametrize(
    "polygon",
    [
        (),
        (WorldPoint(0.0, 0.0),),
        (WorldPoint(0.0, 0.0), WorldPoint(1.0, 1.0), WorldPoint(2.0, 2.0)),
    ],
)
def test_point_margin_rejects_degenerate_hulls(polygon) -> None:
    with pytest.raises(ValueError, match="nondegenerate|three"):
        point_margin_to_convex_polygon(WorldPoint(0.0, 0.0), polygon)


def test_point_margin_rejects_invalid_query_point() -> None:
    square = (
        WorldPoint(0.0, 0.0),
        WorldPoint(1.0, 0.0),
        WorldPoint(1.0, 1.0),
        WorldPoint(0.0, 1.0),
    )

    with pytest.raises(TypeError, match="exact WorldPoint"):
        point_margin_to_convex_polygon((0.5, 0.5), square)
    with pytest.raises(TypeError, match="finite real"):
        point_margin_to_convex_polygon(WorldPoint(False, 0.5), square)
    with pytest.raises(ValueError, match="finite"):
        point_margin_to_convex_polygon(WorldPoint(float("inf"), 0.5), square)


@pytest.mark.parametrize("theta_rad", [0.0, pi / 2.0, 0.37])
def test_oriented_rectangle_matches_wheel_golden_for_representative_poses(
    theta_rad,
) -> None:
    geometry = FineGridGeometryV2(
        width=8,
        height=7,
        origin=(10.0, -3.0),
        frame_id="moon",
    )
    center = WorldPoint(11.25, -1.25)

    cells = oriented_rectangle_cells(center, theta_rad, 0.612, 0.580, geometry)
    wheel_cells = conservative_wheel_pose_cells(
        Pose2D(center.x, center.y, theta_rad),
        geometry,
        body_length_m=0.612,
        body_width_m=0.580,
        safety_margin_m=0.0,
    )

    assert cells == wheel_cells
    assert cells == tuple(sorted(set(cells), key=lambda cell: (cell.y, cell.x)))


def test_oriented_rectangle_uses_closed_cell_contact_without_clamping() -> None:
    geometry = FineGridGeometryV2(width=2, height=2, frame_id="moon")
    dimension = 2.0 * (0.5 - geometry.resolution_m / sqrt(2.0))

    cells = oriented_rectangle_cells(
        WorldPoint(0.25, 0.25),
        0.0,
        dimension,
        dimension,
        geometry,
    )
    negative_cells = oriented_rectangle_cells(
        WorldPoint(-0.10, -0.10),
        pi / 4.0,
        0.60,
        0.40,
        geometry,
    )

    assert Cell(-1, 0) in cells
    assert Cell(0, -1) in cells
    assert any(cell.x < 0 or cell.y < 0 for cell in negative_cells)


def test_oriented_rectangle_checks_candidate_cap_before_enumeration(monkeypatch) -> None:
    def forbidden_range(*_args):
        raise AssertionError("candidate range entered before bound check")

    monkeypatch.setattr(geometry_module, "range", forbidden_range, raising=False)

    with pytest.raises(ValueError, match="candidate.*100000|public bound"):
        oriented_rectangle_cells(
            WorldPoint(1.25, 1.25),
            0.0,
            1.0e9,
            0.40,
            FineGridGeometryV2(width=2, height=2, frame_id="moon"),
        )


def test_oriented_rectangle_requires_exact_center_and_geometry() -> None:
    class DerivedWorldPoint(WorldPoint):
        pass

    class DerivedGeometry(FineGridGeometryV2):
        pass

    geometry = FineGridGeometryV2(width=2, height=2, frame_id="moon")

    with pytest.raises(TypeError, match="exact WorldPoint"):
        oriented_rectangle_cells(
            DerivedWorldPoint(0.25, 0.25), 0.0, 0.60, 0.40, geometry
        )
    with pytest.raises(TypeError, match="exact FineGridGeometryV2"):
        oriented_rectangle_cells(
            WorldPoint(0.25, 0.25),
            0.0,
            0.60,
            0.40,
            DerivedGeometry(width=2, height=2, frame_id="moon"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("center_x", True),
        ("center_y", float("nan")),
        ("theta_rad", True),
        ("theta_rad", float("inf")),
        ("theta_rad", 10**10_000),
        ("length_m", True),
        ("length_m", 0.0),
        ("length_m", float("nan")),
        ("width_m", -0.0),
        ("width_m", float("inf")),
        ("width_m", 10**10_000),
    ],
    ids=[
        "center_x_bool",
        "center_y_nan",
        "theta_bool",
        "theta_inf",
        "theta_huge",
        "length_bool",
        "length_zero",
        "length_nan",
        "width_negative_zero",
        "width_inf",
        "width_huge",
    ],
)
def test_oriented_rectangle_rejects_malicious_numeric_inputs(field, value) -> None:
    center_x = value if field == "center_x" else 0.25
    center_y = value if field == "center_y" else 0.25
    theta_rad = value if field == "theta_rad" else 0.0
    length_m = value if field == "length_m" else 0.60
    width_m = value if field == "width_m" else 0.40

    with pytest.raises((TypeError, ValueError), match="finite|positive|center|theta"):
        oriented_rectangle_cells(
            WorldPoint(center_x, center_y),
            theta_rad,
            length_m,
            width_m,
            FineGridGeometryV2(width=2, height=2, frame_id="moon"),
        )


def test_sample_pose_sweep_handles_zero_move_and_translation_endpoints() -> None:
    start = PoseStateV2(0.0, 0.0, 0.0)

    assert sample_pose_sweep(start, start, 0.25) == (start,)
    assert sample_pose_sweep(start, PoseStateV2(1.0, 0.0, 0.0), 0.4) == (
        start,
        PoseStateV2(1.0 / 3.0, 0.0, 0.0),
        PoseStateV2(2.0 / 3.0, 0.0, 0.0),
        PoseStateV2(1.0, 0.0, 0.0),
    )


def test_sample_pose_sweep_keeps_distinct_equivalent_heading_endpoints() -> None:
    start = PoseStateV2(0.0, 0.0, 0.0)
    end = PoseStateV2(0.0, 0.0, 2.0 * pi)

    assert sample_pose_sweep(start, end, 0.25) == (start, end)


def test_sample_pose_sweep_covers_pure_rotation_and_wraps_shortest_arc() -> None:
    pure_rotation = sample_pose_sweep(
        PoseStateV2(1.0, 2.0, 0.0),
        PoseStateV2(1.0, 2.0, pi / 2.0),
        0.5,
    )
    wrap_start = PoseStateV2(0.0, 0.0, 17.0 * pi / 18.0)
    wrap_end = PoseStateV2(0.0, 0.0, -17.0 * pi / 18.0)
    wrapped = sample_pose_sweep(wrap_start, wrap_end, 0.2)

    assert len(pure_rotation) == 5
    assert pure_rotation[0] == PoseStateV2(1.0, 2.0, 0.0)
    assert pure_rotation[-1] == PoseStateV2(1.0, 2.0, pi / 2.0)
    assert wrapped == (wrap_start, PoseStateV2(0.0, 0.0, pi), wrap_end)


def test_sample_pose_sweep_preserves_positive_and_negative_pi_tie_direction() -> None:
    start = PoseStateV2(0.0, 0.0, 0.0)
    positive = sample_pose_sweep(start, PoseStateV2(0.0, 0.0, pi), pi / 2.0)
    negative = sample_pose_sweep(start, PoseStateV2(0.0, 0.0, -pi), pi / 2.0)

    assert positive == (
        start,
        PoseStateV2(0.0, 0.0, pi / 2.0),
        PoseStateV2(0.0, 0.0, pi),
    )
    assert negative == (
        start,
        PoseStateV2(0.0, 0.0, -pi / 2.0),
        PoseStateV2(0.0, 0.0, -pi),
    )


def test_sample_pose_sweep_checks_cap_before_allocation(monkeypatch) -> None:
    def forbidden_range(*_args):
        raise AssertionError("sample allocation entered before bound check")

    monkeypatch.setattr(geometry_module, "range", forbidden_range, raising=False)

    with pytest.raises(ValueError, match="sample.*100000|public.*cap"):
        sample_pose_sweep(
            PoseStateV2(0.0, 0.0, 0.0),
            PoseStateV2(100_001.0, 0.0, 0.0),
            1.0,
        )


@pytest.mark.parametrize(
    "step_m",
    [True, 0.0, -0.0, -1.0, float("nan"), float("inf"), 10**10_000],
    ids=["bool", "zero", "negative_zero", "negative", "nan", "inf", "huge"],
)
def test_sample_pose_sweep_rejects_invalid_step(step_m) -> None:
    with pytest.raises((TypeError, ValueError), match="step_m.*finite|step_m.*positive"):
        sample_pose_sweep(
            PoseStateV2(0.0, 0.0, 0.0),
            PoseStateV2(1.0, 0.0, 0.0),
            step_m,
        )


def test_sample_pose_sweep_requires_exact_states_and_finite_derived_motion() -> None:
    class DerivedPoseState(PoseStateV2):
        pass

    with pytest.raises(TypeError, match="exact PoseStateV2"):
        sample_pose_sweep(
            DerivedPoseState(0.0, 0.0, 0.0),
            PoseStateV2(1.0, 0.0, 0.0),
            0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        sample_pose_sweep(
            PoseStateV2(-1.0e308, 0.0, 0.0),
            PoseStateV2(1.0e308, 0.0, 0.0),
            0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        sample_pose_sweep(
            PoseStateV2(0.0, 0.0, -1.0e308),
            PoseStateV2(0.0, 0.0, 1.0e308),
            0.5,
        )
