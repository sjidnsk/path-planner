from math import ceil, cos, floor, hypot, isfinite, nextafter, pi, sin, sqrt

import pytest

import path_planner.v2.geometry as geometry_module
from path_planner.core import Cell
from path_planner.search import (
    MotionPrimitive,
    Pose2D,
    PoseTransition,
    replay_motion_primitive,
)
from path_planner.v2.geometry import (
    conservative_wheel_pose_cells,
    conservative_wheel_sweep_cells,
    dense_wheel_replay_step_count,
)
from path_planner.v2.terrain import FineGridGeometryV2


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
