import math

import pytest

from path_planner.platform import PlannerPlatformProfile, load_planner_platform_profile


def test_load_planner_platform_profile_reads_yutu2_from_dev_platform_constraints():
    profile = load_planner_platform_profile(platform="yutu2", safety_margin_m=0.1)

    assert profile.platform_key == "yutu2"
    assert profile.config_path is not None
    assert profile.config_path.name == "yutu2.json"
    assert profile.body_length_m == pytest.approx(1.5)
    assert profile.body_width_m == pytest.approx(1.1)
    assert profile.footprint_radius_m == pytest.approx(math.hypot(1.5, 1.1) / 2.0 + 0.1)
    assert profile.effective_min_turning_radius_m is None
    assert "min_turning_radius" in profile.parameter_sources
    assert any("min_turning_radius" in warning for warning in profile.constraint_warnings)


def test_min_turning_radius_override_is_tracked_as_override_source():
    profile = load_planner_platform_profile(
        platform="yutu2",
        min_turning_radius_override_m=2.5,
    )

    assert profile.raw_min_turning_radius_m == pytest.approx(0.0)
    assert profile.effective_min_turning_radius_m == pytest.approx(2.5)
    assert profile.constraint_sources["effective_min_turning_radius_m"] == "override"


def test_planner_platform_profile_serializes_constraint_sources():
    profile = PlannerPlatformProfile(
        platform_key="test",
        platform_name="Test Platform",
        config_path=None,
        safety_margin_m=0.25,
        body_length_m=2.0,
        body_width_m=1.0,
        footprint_radius_m=1.25,
        max_slope_deg=15.0,
        max_obstacle_height_m=0.2,
        ground_clearance_m=0.3,
        raw_min_turning_radius_m=0.0,
        effective_min_turning_radius_m=None,
        energy_model={"base_cost": 1.0},
        parameter_sources={"body_width": {"source_kind": "estimated", "unit": "m"}},
        constraint_sources={"footprint_radius_m": "derived"},
        constraint_warnings=("min_turning_radius is not enforced",),
    )

    payload = profile.to_dict()

    assert payload["platform_key"] == "test"
    assert payload["derived_constraints"]["footprint_radius_m"] == 1.25
    assert payload["constraint_sources"]["footprint_radius_m"] == "derived"
    assert payload["constraint_warnings"] == ["min_turning_radius is not enforced"]
