from path_planner.core import Cell, WorldPoint
from path_planner.postprocess import (
    CorridorResult,
    CorridorSection,
    CurvatureReport,
    CurvatureSample,
    FallbackStatus,
    PostprocessResult,
    SmoothedPathResult,
)


def test_postprocess_result_serializes_phase2_contract():
    result = PostprocessResult(
        raw_path_cells=(Cell(0, 0), Cell(1, 0), Cell(2, 0)),
        raw_path_world=(WorldPoint(0.0, 0.0), WorldPoint(1.0, 0.0), WorldPoint(2.0, 0.0)),
        corridor=CorridorResult(
            status="ok",
            radius_cells=1,
            sections=(
                CorridorSection(center=Cell(0, 0), cells=(Cell(0, 0), Cell(1, 0))),
            ),
            failure_reason=None,
        ),
        smoothed_path=SmoothedPathResult(
            status="shortcut",
            cells=(Cell(0, 0), Cell(2, 0)),
            world=(WorldPoint(0.0, 0.0), WorldPoint(2.0, 0.0)),
            fallback_reason=None,
        ),
        curvature_report=CurvatureReport(
            is_feasible=True,
            max_curvature=0.0,
            min_turning_radius=None,
            violation_indices=(),
            summary="path satisfies curvature limit",
            samples=(
                CurvatureSample(
                    point_index=1,
                    turn_angle_deg=0.0,
                    curvature=0.0,
                    turning_radius=None,
                    violates=False,
                    violates_curvature=False,
                    violates_min_turning_radius=False,
                ),
            ),
        ),
        fallback_status=FallbackStatus(used_raw_path=False, reason=None),
    )

    payload = result.to_dict()

    assert payload["raw_path"]["cells"] == [[0, 0], [1, 0], [2, 0]]
    assert payload["corridor"]["sections"][0]["center"] == [0, 0]
    assert payload["corridor"]["sections"][0]["bounds"] == {"min": [0, 0], "max": [1, 0]}
    assert payload["smoothed_path"]["cells"] == [[0, 0], [2, 0]]
    assert payload["curvature_report"]["is_feasible"] is True
    assert payload["curvature_report"]["samples"][0]["turn_angle_deg"] == 0.0
    assert payload["fallback_status"] == {"used_raw_path": False, "reason": None}
