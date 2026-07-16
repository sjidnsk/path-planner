from dataclasses import FrozenInstanceError
from math import nextafter, pi

import pytest

from path_planner.v2.contracts import PlatformKindV2
from path_planner.v2.profiles import (
    PLATFORM_PROFILE_SCHEMA_VERSION_V2,
    WHEEL_RELATIVE_ENERGY_PROXY_ID_V2,
    PlatformProfileRegistryV2,
    PlatformProfileV2,
    WheelProfileV2,
)
from path_planner.v2.providers import PrimitiveProviderV2


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
