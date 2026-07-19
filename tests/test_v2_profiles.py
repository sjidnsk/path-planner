from dataclasses import FrozenInstanceError, fields
from math import nextafter, pi

import pytest

import path_planner.v2.profiles as profiles_module
from path_planner.v2.contracts import PlatformKindV2
from path_planner.v2.profiles import (
    HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
    HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2,
    LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2,
    PLATFORM_PROFILE_SCHEMA_VERSION_V2,
    WHEEL_RELATIVE_ENERGY_PROXY_ID_V2,
    HopperProfileAuditV2,
    HopperProfileV2,
    LeggedProfileV2,
    PlatformProfileRegistryV2,
    PlatformProfileV2,
    WheelProfileV2,
    audit_hopper_profile_v2,
)
from path_planner.v2.providers import PrimitiveProviderV2


def test_legged_profile_public_surface_exists() -> None:
    assert (
        profiles_module.LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2
        == "simulation_proxy_static_crawl/v1"
    )
    assert hasattr(profiles_module, "LeggedProfileV2")


def _profile(profile_id: str = "wheel-safe/v1", **overrides) -> PlatformProfileV2:
    values = {
        "profile_id": profile_id,
        "platform_kind": PlatformKindV2.WHEEL,
        "capability_revision": "wheel-capability/v1",
        "simulation_proxy": False,
        "max_traversable_slope_deg": 30.0,
    }
    values.update(overrides)
    return PlatformProfileV2(**values)


def _legged_platform(**overrides) -> PlatformProfileV2:
    values = {
        "profile_id": "legged-static-crawl/v1",
        "platform_kind": PlatformKindV2.LEGGED,
        "capability_revision": LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2,
        "simulation_proxy": True,
        "max_traversable_slope_deg": 30.0,
        "goal_position_tolerance_m": 0.0,
        "goal_heading_tolerance_rad": 0.0,
    }
    values.update(overrides)
    return PlatformProfileV2(**values)


def _hopper_platform(**overrides) -> PlatformProfileV2:
    values = {
        "profile_id": "hopper-lunar-ballistic-proxy/v1",
        "platform_kind": PlatformKindV2.HOPPER,
        "capability_revision": HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
        "simulation_proxy": True,
        "max_traversable_slope_deg": 30.0,
        "goal_position_tolerance_m": 0.0,
        "goal_heading_tolerance_rad": 0.0,
    }
    values.update(overrides)
    return PlatformProfileV2(**values)


def _complete_hopper(**overrides) -> HopperProfileV2:
    values = {
        "profile": _hopper_platform(),
        "body_envelope_radius_m": 0.25,
        "launch_reference_height_m": 0.50,
        "arc_clearance_margin_m": 0.10,
        "landing_footprint_radius_m": 0.30,
        "stop_condition": "fixture_stop_proxy/v1",
        "energy_model": "fixture_energy_proxy/v1",
    }
    values.update(overrides)
    return HopperProfileV2(**values)


def test_platform_profile_is_frozen_slotted_and_has_fixed_schema() -> None:
    profile = _profile()

    assert profile.schema_version == "path-planner-v2-platform-profile/v1"
    assert profile.schema_version == PLATFORM_PROFILE_SCHEMA_VERSION_V2
    assert not hasattr(profile, "__dict__")
    with pytest.raises(FrozenInstanceError):
        profile.profile_id = "other"
    with pytest.raises(ValueError, match="schema_version"):
        _profile(schema_version="other")


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"profile_id": " "}, ValueError, "profile_id"),
        ({"profile_id": 1}, ValueError, "profile_id"),
        ({"capability_revision": ""}, ValueError, "capability_revision"),
        ({"platform_kind": "wheel"}, TypeError, "PlatformKindV2"),
        ({"simulation_proxy": 0}, TypeError, "bool"),
        ({"max_traversable_slope_deg": True}, TypeError, "finite"),
        ({"max_traversable_slope_deg": float("nan")}, ValueError, "finite"),
        ({"max_traversable_slope_deg": float("inf")}, ValueError, "finite"),
        ({"max_traversable_slope_deg": -0.1}, ValueError, r"\[0, 90\]"),
        ({"max_traversable_slope_deg": 90.0001}, ValueError, r"\[0, 90\]"),
        ({"goal_position_tolerance_m": True}, TypeError, "finite"),
        ({"goal_position_tolerance_m": float("nan")}, ValueError, "finite"),
        ({"goal_position_tolerance_m": -0.1}, ValueError, "nonnegative"),
        ({"goal_heading_tolerance_rad": True}, TypeError, "finite"),
        ({"goal_heading_tolerance_rad": float("inf")}, ValueError, "finite"),
        ({"goal_heading_tolerance_rad": -0.1}, ValueError, "nonnegative"),
        ({"goal_heading_tolerance_rad": nextafter(pi, float("inf"))}, ValueError, "pi"),
    ],
)
def test_platform_profile_rejects_invalid_identity_and_safety_values(
    overrides,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        _profile(**overrides)


def test_platform_profile_goal_tolerance_defaults_and_closed_boundaries() -> None:
    default = _profile()
    boundary = _profile(
        goal_position_tolerance_m=0.25,
        goal_heading_tolerance_rad=pi,
    )

    assert default.goal_position_tolerance_m == 0.0
    assert default.goal_heading_tolerance_rad == 0.0
    assert boundary.goal_position_tolerance_m == 0.25
    assert boundary.goal_heading_tolerance_rad == pi


def test_registry_requires_sorted_unique_exact_profiles_and_resolves_without_fallback() -> None:
    alpha = _profile("alpha/v1")
    beta = _profile("beta/v1", platform_kind=PlatformKindV2.LEGGED)
    registry = PlatformProfileRegistryV2((alpha, beta))

    assert registry.profiles == (alpha, beta)
    assert registry.resolve("alpha/v1") is alpha
    assert registry.resolve("unknown/v1") is None
    assert not hasattr(registry, "__dict__")
    with pytest.raises(FrozenInstanceError):
        registry.profiles = ()
    with pytest.raises(TypeError, match="tuple"):
        PlatformProfileRegistryV2([alpha])
    with pytest.raises(TypeError, match="PlatformProfileV2"):
        PlatformProfileRegistryV2((object(),))
    with pytest.raises(ValueError, match="sorted"):
        PlatformProfileRegistryV2((beta, alpha))
    with pytest.raises(ValueError, match="unique"):
        PlatformProfileRegistryV2((alpha, alpha))
    with pytest.raises(TypeError, match="profile_id"):
        registry.resolve(1)


def test_provider_protocol_is_runtime_checkable_and_exposes_profile_and_plan() -> None:
    profile = _profile()

    class Provider:
        def __init__(self) -> None:
            self.profile = profile

        def plan(self, request, anchor, deadline):
            raise NotImplementedError

    provider = Provider()

    assert isinstance(provider, PrimitiveProviderV2)
    assert provider.profile is profile
    assert callable(provider.plan)


def test_wheel_profile_freezes_platform_geometry_and_relative_energy_contract() -> None:
    profile = _profile()
    wheel = WheelProfileV2(profile=profile)

    assert wheel.profile is profile
    assert wheel.steering_model == "differential_skid_steer"
    assert (wheel.body_length_m, wheel.body_width_m) == (0.612, 0.580)
    assert wheel.profile.max_traversable_slope_deg == 30.0
    assert wheel.reverse_enabled is True
    assert wheel.turn_in_place_enabled is True
    assert wheel.relative_energy_proxy_id == WHEEL_RELATIVE_ENERGY_PROXY_ID_V2
    assert wheel.energy_normalization > 0.0
    assert wheel.time_normalization_s > 0.0
    assert not hasattr(wheel, "__dict__")
    with pytest.raises(FrozenInstanceError):
        wheel.body_length_m = 1.0


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"profile": _profile(platform_kind=PlatformKindV2.LEGGED)}, ValueError, "wheel"),
        (
            {"profile": _profile(max_traversable_slope_deg=nextafter(30.0, float("inf")))},
            ValueError,
            "30.0",
        ),
        ({"steering_model": "ackermann"}, ValueError, "steering_model"),
        ({"body_length_m": 0.611}, ValueError, "0.612"),
        ({"body_width_m": 0.581}, ValueError, "0.580"),
        ({"footprint_safety_margin_m": True}, TypeError, "finite"),
        ({"footprint_safety_margin_m": -0.1}, ValueError, "nonnegative"),
        ({"reverse_enabled": 1}, TypeError, "bool"),
        ({"turn_in_place_enabled": 0}, TypeError, "bool"),
        ({"min_turning_radius_m": float("nan")}, ValueError, "finite"),
        ({"theta_bin_count": True}, TypeError, "integer"),
        ({"theta_bin_count": 0}, ValueError, "positive"),
        ({"primitive_duration_s": 0.0}, ValueError, "positive"),
        ({"integration_dt_s": float("inf")}, ValueError, "finite"),
        ({"max_speed_mps": -1.0}, ValueError, "positive"),
        ({"max_angular_speed_radps": 0.0}, ValueError, "positive"),
        ({"relative_energy_proxy_id": "unversioned"}, ValueError, "versioned"),
        ({"relative_energy_proxy_id": "physical_energy/v1"}, ValueError, "relative-energy"),
        ({"translation_energy_per_m": -0.1}, ValueError, "nonnegative"),
        ({"rotation_energy_per_rad": True}, TypeError, "finite"),
        ({"idle_energy_per_s": float("nan")}, ValueError, "finite"),
        ({"reverse_energy_multiplier": -0.1}, ValueError, "nonnegative"),
        ({"energy_normalization": 0.0}, ValueError, "positive"),
        ({"time_normalization_s": 0.0}, ValueError, "positive"),
    ],
)
def test_wheel_profile_rejects_inexact_or_nonfinite_contracts(
    overrides,
    error,
    message,
) -> None:
    values = {"profile": _profile()}
    values.update(overrides)

    with pytest.raises(error, match=message):
        WheelProfileV2(**values)


def test_profile_numeric_helpers_convert_huge_real_overflow_to_finite_errors() -> None:
    huge = 10**10_000

    with pytest.raises(ValueError, match="max_traversable_slope_deg.*finite"):
        _profile(max_traversable_slope_deg=huge)
    with pytest.raises(ValueError, match="goal_position_tolerance_m.*finite"):
        _profile(goal_position_tolerance_m=huge)
    with pytest.raises(ValueError, match="energy_normalization.*finite"):
        WheelProfileV2(profile=_profile(), energy_normalization=huge)
    with pytest.raises(ValueError, match="translation_energy_per_m.*finite"):
        WheelProfileV2(profile=_profile(), translation_energy_per_m=huge)


def test_legged_profile_freezes_static_crawl_proxy_contract() -> None:
    profile = _legged_platform()
    legged = LeggedProfileV2(profile=profile)

    assert legged.profile is profile
    assert legged.profile.platform_kind is PlatformKindV2.LEGGED
    assert legged.profile.simulation_proxy is True
    assert (
        legged.profile.capability_revision
        == LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2
        == "simulation_proxy_static_crawl/v1"
    )
    assert legged.profile.max_traversable_slope_deg == 30.0
    assert legged.profile.goal_position_tolerance_m == 0.0
    assert legged.profile.goal_heading_tolerance_rad == 0.0
    assert legged.body_length_m == 0.60
    assert legged.body_width_m == 0.40
    assert legged.nominal_foot_rectangle_length_m == 0.70
    assert legged.nominal_foot_rectangle_width_m == 0.50
    assert legged.max_step_length_m == 0.50
    assert legged.max_step_height_m == 0.25
    assert legged.max_foothold_slope_deg == 25.0
    assert legged.min_support_margin_m == 0.05
    assert legged.local_foothold_grid_spacing_m == 0.25
    assert not hasattr(legged, "__dict__")
    with pytest.raises(FrozenInstanceError):
        legged.body_length_m = 1.0


def test_legged_profile_accepts_signed_zero_goal_tolerances() -> None:
    profile = _legged_platform(
        goal_position_tolerance_m=-0.0,
        goal_heading_tolerance_rad=-0.0,
    )

    assert LeggedProfileV2(profile=profile).profile is profile


def test_legged_profile_requires_exact_platform_profile() -> None:
    class DerivedPlatformProfile(PlatformProfileV2):
        pass

    derived = DerivedPlatformProfile(
        profile_id="legged-static-crawl/v1",
        platform_kind=PlatformKindV2.LEGGED,
        capability_revision=LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2,
        simulation_proxy=True,
        max_traversable_slope_deg=30.0,
    )

    with pytest.raises(TypeError, match="exact PlatformProfileV2"):
        LeggedProfileV2(profile=derived)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (_profile(), "LEGGED"),
        (_legged_platform(simulation_proxy=False), "simulation_proxy"),
        (_legged_platform(capability_revision="simulation_proxy_static_crawl/v2"), "capability_revision"),
        (
            _legged_platform(
                max_traversable_slope_deg=nextafter(30.0, float("inf"))
            ),
            "30.0",
        ),
        (
            _legged_platform(goal_position_tolerance_m=nextafter(0.0, 1.0)),
            "goal_position_tolerance_m",
        ),
        (
            _legged_platform(goal_heading_tolerance_rad=nextafter(0.0, 1.0)),
            "goal_heading_tolerance_rad",
        ),
    ],
)
def test_legged_profile_rejects_incompatible_base_profile(profile, message) -> None:
    with pytest.raises(ValueError, match=message):
        LeggedProfileV2(profile=profile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("body_length_m", 0.60),
        ("body_width_m", 0.40),
        ("nominal_foot_rectangle_length_m", 0.70),
        ("nominal_foot_rectangle_width_m", 0.50),
        ("max_step_length_m", 0.50),
        ("max_step_height_m", 0.25),
        ("max_foothold_slope_deg", 25.0),
        ("min_support_margin_m", 0.05),
        ("local_foothold_grid_spacing_m", 0.25),
    ],
)
def test_legged_profile_rejects_every_frozen_numeric_deviation(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        LeggedProfileV2(
            profile=_legged_platform(),
            **{field: nextafter(value, float("inf"))},
        )


@pytest.mark.parametrize(
    "field",
    [
        "body_length_m",
        "body_width_m",
        "nominal_foot_rectangle_length_m",
        "nominal_foot_rectangle_width_m",
        "max_step_length_m",
        "max_step_height_m",
        "max_foothold_slope_deg",
        "min_support_margin_m",
        "local_foothold_grid_spacing_m",
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), float("-inf"), -0.0, 10**10_000],
    ids=["bool", "nan", "positive_inf", "negative_inf", "signed_zero", "huge"],
)
def test_legged_profile_rejects_malicious_frozen_numeric_values(field, value) -> None:
    with pytest.raises((TypeError, ValueError), match=f"{field}|finite"):
        LeggedProfileV2(profile=_legged_platform(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("goal_position_tolerance_m", False, TypeError, "goal_position_tolerance_m.*exact.*float"),
        ("max_traversable_slope_deg", 30, TypeError, "max_traversable_slope_deg.*exact.*float"),
        ("goal_heading_tolerance_rad", 0, TypeError, "goal_heading_tolerance_rad.*exact.*float"),
        ("profile_id", "", ValueError, "profile_id"),
        ("schema_version", "forged-schema/v1", ValueError, "schema_version"),
    ],
)
def test_legged_profile_reaudits_forged_base_profile_fields(
    field,
    value,
    error,
    message,
) -> None:
    profile = _legged_platform()
    object.__setattr__(profile, field, value)

    with pytest.raises(error, match=message):
        LeggedProfileV2(profile=profile)


@pytest.mark.parametrize("field", ["profile_id", "capability_revision", "schema_version"])
def test_legged_profile_rejects_forged_string_subclasses(field) -> None:
    class DerivedStr(str):
        pass

    profile = _legged_platform()
    object.__setattr__(profile, field, DerivedStr(getattr(profile, field)))

    with pytest.raises(TypeError, match=f"{field}.*exact.*str"):
        LeggedProfileV2(profile=profile)


def test_legged_profile_numeric_conversion_stabilizes_runtime_error() -> None:
    class ExplodingInt(int):
        def __float__(self):
            raise RuntimeError("numeric protocol exploded")

    with pytest.raises(ValueError, match="body_length_m.*finite"):
        LeggedProfileV2(
            profile=_legged_platform(),
            body_length_m=ExplodingInt(1),
        )


def test_hopper_profile_freezes_approved_proxy_defaults() -> None:
    hopper = HopperProfileV2(profile=_hopper_platform())
    assert tuple(field.name for field in fields(HopperProfileV2)) == (
        "profile",
        "gravity_mps2",
        "launch_speeds_mps",
        "launch_elevations_rad",
        "azimuth_direction_count",
        "landing_sigma_range_scale",
        "landing_sigma_offset_m",
        "max_landing_slope_deg",
        "landing_probability_threshold",
        "midcourse_correction_enabled",
        "inflight_observation_enabled",
        "body_envelope_radius_m",
        "launch_reference_height_m",
        "arc_clearance_margin_m",
        "landing_footprint_radius_m",
        "stop_condition",
        "energy_model",
    )
    assert HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2 == (
        "simulation_proxy_lunar_ballistic/v1"
    )
    assert HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2 == (
        "hopper_proxy_profile_incomplete"
    )
    assert hopper.profile.platform_kind is PlatformKindV2.HOPPER
    assert hopper.profile.simulation_proxy is True
    assert hopper.profile.max_traversable_slope_deg == 30.0
    assert hopper.profile.goal_position_tolerance_m == 0.0
    assert hopper.profile.goal_heading_tolerance_rad == 0.0
    assert hopper.gravity_mps2 == 1.62
    assert hopper.launch_speeds_mps == (1.5, 2.0, 2.5, 3.0)
    assert hopper.launch_elevations_rad == (pi / 6.0, pi / 4.0, pi / 3.0)
    assert hopper.azimuth_direction_count == 16
    assert hopper.landing_sigma_range_scale == 0.05
    assert hopper.landing_sigma_offset_m == 0.05
    assert hopper.max_landing_slope_deg == 15.0
    assert hopper.landing_probability_threshold == 0.99
    assert hopper.midcourse_correction_enabled is False
    assert hopper.inflight_observation_enabled is False
    assert not hasattr(hopper, "__dict__")
    with pytest.raises(FrozenInstanceError):
        hopper.gravity_mps2 = 1.0


def test_hopper_profile_keeps_formal_capability_fields_explicitly_nullable() -> None:
    hopper = HopperProfileV2(profile=_hopper_platform())
    assert (
        hopper.body_envelope_radius_m,
        hopper.launch_reference_height_m,
        hopper.arc_clearance_margin_m,
        hopper.landing_footprint_radius_m,
        hopper.stop_condition,
        hopper.energy_model,
    ) == (None, None, None, None, None, None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"platform_kind": PlatformKindV2.LEGGED}, "HOPPER"),
        ({"simulation_proxy": False}, "simulation_proxy"),
        ({"capability_revision": "simulation_proxy_lunar_ballistic/v2"}, "capability_revision"),
        ({"max_traversable_slope_deg": nextafter(30.0, float("inf"))}, "30.0"),
        ({"goal_position_tolerance_m": nextafter(0.0, 1.0)}, "goal_position_tolerance_m"),
        ({"goal_heading_tolerance_rad": nextafter(0.0, 1.0)}, "goal_heading_tolerance_rad"),
    ],
)
def test_hopper_profile_rejects_incompatible_base_profile(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        HopperProfileV2(profile=_hopper_platform(**overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gravity_mps2", nextafter(1.62, float("inf"))),
        ("launch_speeds_mps", (1.5, 2.0, 2.5, nextafter(3.0, float("inf")))),
        ("launch_elevations_rad", (pi / 6.0, pi / 4.0, nextafter(pi / 3.0, float("inf")))),
        ("azimuth_direction_count", 15),
        ("landing_sigma_range_scale", nextafter(0.05, float("inf"))),
        ("landing_sigma_offset_m", nextafter(0.05, float("inf"))),
        ("max_landing_slope_deg", nextafter(15.0, float("inf"))),
        ("landing_probability_threshold", nextafter(0.99, float("inf"))),
        ("midcourse_correction_enabled", True),
        ("inflight_observation_enabled", True),
    ],
)
def test_hopper_profile_rejects_every_frozen_proxy_field_deviation(field, value) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        HopperProfileV2(profile=_hopper_platform(), **{field: value})


def test_hopper_profile_rejects_inexact_containers_elements_and_switches() -> None:
    class DerivedFloat(float):
        pass

    with pytest.raises(TypeError, match="gravity_mps2.*exact.*float"):
        HopperProfileV2(profile=_hopper_platform(), gravity_mps2=DerivedFloat(1.62))
    with pytest.raises(TypeError, match="launch_speeds_mps.*exact tuple"):
        HopperProfileV2(profile=_hopper_platform(), launch_speeds_mps=[1.5, 2.0, 2.5, 3.0])
    with pytest.raises(TypeError, match="launch_speeds_mps.*exact float"):
        HopperProfileV2(profile=_hopper_platform(), launch_speeds_mps=(1.5, 2.0, 2.5, 3))
    with pytest.raises(TypeError, match="azimuth_direction_count.*exact int"):
        HopperProfileV2(profile=_hopper_platform(), azimuth_direction_count=True)
    with pytest.raises(TypeError, match="midcourse_correction_enabled.*exact bool"):
        HopperProfileV2(profile=_hopper_platform(), midcourse_correction_enabled=0)


def test_hopper_profile_validates_explicit_formal_capability_fields() -> None:
    hopper = HopperProfileV2(
        profile=_hopper_platform(),
        body_envelope_radius_m=0.25,
        launch_reference_height_m=0.50,
        arc_clearance_margin_m=0.10,
        landing_footprint_radius_m=0.30,
        stop_condition="fixture_stop_proxy/v1",
        energy_model="fixture_energy_proxy/v1",
    )
    assert hopper.body_envelope_radius_m == 0.25
    assert hopper.stop_condition == "fixture_stop_proxy/v1"

    invalid = (
        ("body_envelope_radius_m", 0.0),
        ("launch_reference_height_m", 1),
        ("arc_clearance_margin_m", True),
        ("landing_footprint_radius_m", float("nan")),
        ("stop_condition", "unversioned"),
        ("energy_model", "fixture/v0"),
    )
    for field, value in invalid:
        with pytest.raises((TypeError, ValueError), match=field):
            HopperProfileV2(profile=_hopper_platform(), **{field: value})


def test_hopper_profile_reaudits_forged_exact_base_profile() -> None:
    class DerivedPlatformProfile(PlatformProfileV2):
        pass

    derived = DerivedPlatformProfile(
        profile_id="hopper-derived/v1",
        platform_kind=PlatformKindV2.HOPPER,
        capability_revision=HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
        simulation_proxy=True,
        max_traversable_slope_deg=30.0,
    )
    with pytest.raises(TypeError, match="exact PlatformProfileV2"):
        HopperProfileV2(profile=derived)

    profile = _hopper_platform()
    object.__setattr__(profile, "max_traversable_slope_deg", True)
    with pytest.raises(TypeError, match="max_traversable_slope_deg.*exact.*float"):
        HopperProfileV2(profile=profile)


@pytest.mark.parametrize(
    "field",
    (
        "profile_id",
        "platform_kind",
        "capability_revision",
        "simulation_proxy",
        "max_traversable_slope_deg",
        "goal_position_tolerance_m",
        "goal_heading_tolerance_rad",
        "schema_version",
    ),
)
def test_hopper_profile_rejects_missing_required_base_profile_fields(field) -> None:
    profile = _hopper_platform()
    object.__delattr__(profile, field)

    with pytest.raises((TypeError, ValueError), match=field):
        HopperProfileV2(profile=profile)


HOPPER_MISSING_FIELDS = (
    "body_envelope_radius_m",
    "launch_reference_height_m",
    "arc_clearance_margin_m",
    "landing_footprint_radius_m",
    "stop_condition",
    "energy_model",
)


def test_hopper_profile_audit_reports_all_missing_fields_in_canonical_order() -> None:
    audit = audit_hopper_profile_v2(HopperProfileV2(profile=_hopper_platform()))
    assert tuple(field.name for field in fields(HopperProfileAuditV2)) == (
        "complete", "reason_code", "missing_fields"
    )
    assert audit == HopperProfileAuditV2(
        complete=False,
        reason_code=HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2,
        missing_fields=HOPPER_MISSING_FIELDS,
    )
    assert not hasattr(audit, "__dict__")
    with pytest.raises(FrozenInstanceError):
        audit.complete = True


@pytest.mark.parametrize("missing_field", HOPPER_MISSING_FIELDS)
def test_hopper_profile_audit_reports_each_actual_missing_subset(missing_field) -> None:
    audit = audit_hopper_profile_v2(_complete_hopper(**{missing_field: None}))
    assert audit.complete is False
    assert audit.reason_code == HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2
    assert audit.missing_fields == (missing_field,)


def test_hopper_profile_audit_reports_structural_completion_only() -> None:
    profile = _complete_hopper(
        stop_condition="unsupported_but_structurally_versioned/v7",
        energy_model="unsupported_but_structurally_versioned/v9",
    )
    assert audit_hopper_profile_v2(profile) == HopperProfileAuditV2(True, None, ())


def test_hopper_profile_audit_reaudits_forged_exact_profile() -> None:
    profile = _complete_hopper()
    object.__setattr__(profile, "gravity_mps2", nextafter(1.62, float("inf")))
    with pytest.raises(ValueError, match="gravity_mps2"):
        audit_hopper_profile_v2(profile)


def test_hopper_profile_audit_stabilizes_deleted_required_profile_field() -> None:
    profile = _complete_hopper()
    object.__delattr__(profile, "stop_condition")

    with pytest.raises((TypeError, ValueError), match="stop_condition"):
        audit_hopper_profile_v2(profile)


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ((1, None, ()), TypeError),
        ((True, "hopper_proxy_profile_incomplete", ()), ValueError),
        ((False, None, ("energy_model",)), ValueError),
        ((False, "hopper_proxy_profile_incomplete", ()), ValueError),
        ((False, "hopper_proxy_profile_incomplete", ["energy_model"]), TypeError),
        ((False, "hopper_proxy_profile_incomplete", ("unknown",)), ValueError),
        (
            (
                False,
                "hopper_proxy_profile_incomplete",
                ("energy_model", "body_envelope_radius_m"),
            ),
            ValueError,
        ),
    ],
)
def test_hopper_profile_audit_rejects_inconsistent_contracts(values, error) -> None:
    with pytest.raises(error):
        HopperProfileAuditV2(*values)


def test_hopper_profile_audit_requires_exact_hopper_profile() -> None:
    with pytest.raises(TypeError, match="exact HopperProfileV2"):
        audit_hopper_profile_v2(object())
