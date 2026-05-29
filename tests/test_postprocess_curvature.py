import math

from path_planner.core import WorldPoint
from path_planner.postprocess import check_curvature


def test_check_curvature_marks_straight_path_feasible():
    report = check_curvature(
        (WorldPoint(0.0, 0.0), WorldPoint(1.0, 0.0), WorldPoint(2.0, 0.0)),
        max_curvature=0.5,
    )

    assert report.is_feasible is True
    assert report.max_curvature == 0.0
    assert report.min_turning_radius is None
    assert report.violation_indices == ()


def test_check_curvature_accepts_turn_under_limit():
    report = check_curvature(
        (WorldPoint(0.0, 0.0), WorldPoint(1.0, 0.0), WorldPoint(2.0, 1.0)),
        max_curvature=2.0,
    )

    assert report.is_feasible is True
    assert report.max_curvature > 0.0
    assert report.min_turning_radius is not None
    assert report.violation_indices == ()
    assert report.samples[0].point_index == 1
    assert math.isclose(report.samples[0].turn_angle_deg, 45.0, rel_tol=1e-9)
    assert report.samples[0].turning_radius is not None
    assert report.samples[0].violates is False


def test_check_curvature_reports_sharp_turn_violation():
    report = check_curvature(
        (WorldPoint(0.0, 0.0), WorldPoint(1.0, 0.0), WorldPoint(1.0, 1.0)),
        max_curvature=1.0,
    )

    assert report.is_feasible is False
    assert math.isclose(report.max_curvature, math.sqrt(2.0), rel_tol=1e-9)
    assert report.violation_indices == (1,)
    assert report.summary == "curvature violations: 1"
    assert math.isclose(report.samples[0].turn_angle_deg, 90.0, rel_tol=1e-9)
    assert report.samples[0].violates is True


def test_check_curvature_reports_min_turning_radius_violation():
    report = check_curvature(
        (WorldPoint(0.0, 0.0), WorldPoint(1.0, 0.0), WorldPoint(1.0, 1.0)),
        max_curvature=2.0,
        min_turning_radius=1.0,
    )

    assert report.is_feasible is False
    assert report.violation_indices == (1,)
    assert report.samples[0].violates is True
    assert report.samples[0].violates_min_turning_radius is True
