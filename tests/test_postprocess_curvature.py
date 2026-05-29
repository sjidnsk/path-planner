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


def test_check_curvature_reports_sharp_turn_violation():
    report = check_curvature(
        (WorldPoint(0.0, 0.0), WorldPoint(1.0, 0.0), WorldPoint(1.0, 1.0)),
        max_curvature=1.0,
    )

    assert report.is_feasible is False
    assert math.isclose(report.max_curvature, math.sqrt(2.0), rel_tol=1e-9)
    assert report.violation_indices == (1,)
    assert report.summary == "curvature violations: 1"
