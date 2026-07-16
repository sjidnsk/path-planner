from dataclasses import FrozenInstanceError, fields
from importlib import import_module

import pytest


def _v2():
    return import_module("path_planner.v2")


def _primitive(v2, *, kind=None, start_state=None, end_state=None, **overrides):
    values = {
        "kind": kind or v2.PrimitiveKindV2.WHEEL_MOTION,
        "start_state": start_state or v2.PoseStateV2(0.0, 0.0, 0.0),
        "end_state": end_state or v2.PoseStateV2(1.0, 2.0, 0.5),
        "duration_s": 1.25,
        "distance_m": 2.5,
        "energy_cost": 0.5,
        "observation_contribution": 4.0,
        "validation_level": v2.ValidationLevelV2.L2,
    }
    values.update(overrides)
    return v2.RoutePrimitiveV2(**values)


def _route(v2, *, platform_kind=None, primitives=None, is_complete: bool = True, total_cost=3.5):
    platform_kind = platform_kind or v2.PlatformKindV2.WHEEL
    if primitives is None:
        kind_by_platform = {
            v2.PlatformKindV2.WHEEL: v2.PrimitiveKindV2.WHEEL_MOTION,
            v2.PlatformKindV2.LEGGED: v2.PrimitiveKindV2.LEG_STEP,
            v2.PlatformKindV2.HOPPER: v2.PrimitiveKindV2.BALLISTIC_JUMP,
        }
        primitives = (_primitive(v2, kind=kind_by_platform[platform_kind]),)
    return v2.TypedRouteV2(
        platform_kind=platform_kind,
        primitives=primitives,
        total_cost=total_cost,
        is_complete=is_complete,
    )


def _evidence(v2, *, level=None, passed: bool = True):
    return v2.ValidationEvidenceV2(
        validator_id="route-safety/v1",
        level=level or v2.ValidationLevelV2.L2,
        passed=passed,
        checks=("finite", "collision-free"),
    )


def _projection(v2):
    return v2.ObservationProjectionV2(
        source="terrain-observation/v1",
        sample_states=(v2.PoseStateV2(0.0, 0.0, 0.0), v2.PoseStateV2(1.0, 2.0, 0.5)),
        expected_new_observed_cells=4.0,
        expected_information_gain=0.75,
    )


def _cost(v2, *, total_cost=3.5):
    return v2.CostBreakdownV2(
        distance_cost=1.0,
        risk_cost=0.5,
        energy_cost=0.75,
        time_cost=1.25,
        total_cost=total_cost,
    )


def _telemetry(v2):
    return v2.SearchTelemetryV2(
        expanded_states=8,
        generated_primitives=3,
        rejected_l0=1,
        rejected_l1=0,
        rejected_l2=0,
        elapsed_s=0.25,
        timed_out=False,
        accelerator_used=True,
        ackermann_feasible_claimed=False,
        termination_reason="goal_reached",
    )


def _cache(v2):
    return v2.CacheEvidenceV2(cache_namespace="planner-v2", cache_key="request-001", hit=False)


def _failure_evidence(v2):
    return v2.FailureEvidenceV2(
        stage="search",
        checks=("frontier-exhausted",),
        details=(("expanded", 8), ("last_candidate", None), ("retryable", False)),
    )


def _success(v2, *, platform_kind=None, route=None, cost_breakdown=None, validation=None):
    platform_kind = platform_kind or v2.PlatformKindV2.WHEEL
    return v2.PlanningSuccessV2(
        request_id="request-001",
        platform_kind=platform_kind,
        route=route if route is not None else _route(v2, platform_kind=platform_kind),
        observation_projection=_projection(v2),
        cost_breakdown=cost_breakdown if cost_breakdown is not None else _cost(v2),
        validation_evidence=validation if validation is not None else _evidence(v2),
        search_telemetry=_telemetry(v2),
        cache_evidence=_cache(v2),
    )


def _failure(v2, *, platform_kind=None):
    evidence = _failure_evidence(v2)
    if platform_kind is None:
        evidence = v2.FailureEvidenceV2(
            stage="profile_resolution",
            checks=("platform-kind-unresolved",),
            details=(("profile_id", "unknown"),),
        )
    return v2.PlanningFailureV2(
        request_id="request-001",
        platform_kind=platform_kind,
        category=v2.FailureCategoryV2.NO_COMPLETE_ROUTE,
        reason_code="search_exhausted",
        evidence=evidence,
        search_telemetry=_telemetry(v2),
    )


def test_failure_category_values_are_stable_and_ordered():
    v2 = _v2()

    assert v2.FailureCategoryV2.values() == (
        "invalid_request",
        "unsupported_capability",
        "unsafe_start",
        "unsafe_goal",
        "goal_pose_unreachable",
        "no_complete_route",
        "validation_failed",
        "resource_limit",
        "timeout",
        "internal_error",
    )
    assert tuple(item.value for item in v2.PlatformKindV2) == (
        "wheel",
        "legged",
        "hopper",
    )
    assert v2.PrimitiveKindV2.values() == (
        "wheel_motion",
        "leg_step",
        "ballistic_jump",
    )
    assert v2.ValidationLevelV2.values() == ("L0", "L1", "L2")


def test_result_contracts_have_exact_public_field_names():
    v2 = _v2()

    assert tuple(field.name for field in fields(v2.PlanningSuccessV2)) == (
        "request_id",
        "platform_kind",
        "route",
        "observation_projection",
        "cost_breakdown",
        "validation_evidence",
        "search_telemetry",
        "cache_evidence",
        "schema_version",
    )
    assert tuple(field.name for field in fields(v2.PlanningFailureV2)) == (
        "request_id",
        "platform_kind",
        "category",
        "reason_code",
        "evidence",
        "search_telemetry",
        "schema_version",
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_pose_state_rejects_non_finite_values_and_bool(value):
    v2 = _v2()

    with pytest.raises((TypeError, ValueError), match="finite"):
        v2.PoseStateV2(value, 0.0, 0.0)


def test_pose_state_is_frozen():
    v2 = _v2()
    pose = v2.PoseStateV2(1.0, 2.0, 0.25)

    with pytest.raises(FrozenInstanceError):
        pose.x_m = 9.0


def test_objective_and_resource_budget_fail_closed():
    v2 = _v2()

    with pytest.raises(ValueError, match="nonnegative"):
        v2.ObjectiveProfileV2(distance_weight=-1.0)
    with pytest.raises(TypeError, match="finite"):
        v2.ObjectiveProfileV2(risk_weight=True)
    with pytest.raises(TypeError, match="integer"):
        v2.ResourceBudgetV2(max_expanded_states=True)
    with pytest.raises(ValueError, match="nonnegative"):
        v2.ResourceBudgetV2(max_route_states=-1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"request_id": " "}, "request_id"),
        ({"platform_profile_id": ""}, "platform_profile_id"),
        ({"timeout_s": float("inf")}, "finite"),
        ({"timeout_s": -0.1}, "nonnegative"),
        ({"timeout_s": True}, "finite"),
        ({"determinism_seed": True}, "integer"),
    ],
)
def test_planning_request_validates_identifiers_time_and_seed(overrides, message):
    v2 = _v2()
    fields = {
        "request_id": "request-001",
        "platform_profile_id": "wheel-profile/v1",
        "start_state": v2.PoseStateV2(0.0, 0.0, 0.0),
        "goal_state": v2.PoseStateV2(1.0, 2.0, 0.5),
        "terrain_snapshot": object(),
        "objective_profile": v2.ObjectiveProfileV2(),
        "resource_budget": v2.ResourceBudgetV2(),
        "timeout_s": 10.0,
        "accelerator_policy": v2.AcceleratorPolicyV2.OPTIONAL,
        "determinism_seed": 7,
    }
    fields.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        v2.PlanningRequestV2(**fields)


@pytest.mark.parametrize(
    ("platform_kind", "primitive_kind"),
    [
        ("WHEEL", "WHEEL_MOTION"),
        ("LEGGED", "LEG_STEP"),
        ("HOPPER", "BALLISTIC_JUMP"),
    ],
)
def test_typed_route_accepts_each_platforms_primitive(platform_kind, primitive_kind):
    v2 = _v2()
    platform = getattr(v2.PlatformKindV2, platform_kind)
    kind = getattr(v2.PrimitiveKindV2, primitive_kind)

    route = _route(v2, platform_kind=platform, primitives=(_primitive(v2, kind=kind),))

    assert route.platform_kind is platform
    assert route.primitives[0].kind is kind


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("duration_s", float("nan"), "finite"),
        ("distance_m", -1.0, "nonnegative"),
        ("energy_cost", True, "finite"),
        ("observation_contribution", float("inf"), "finite"),
    ],
)
def test_route_primitive_rejects_invalid_numeric_values(field, value, message):
    v2 = _v2()

    with pytest.raises((TypeError, ValueError), match=message):
        _primitive(v2, **{field: value})


def test_route_primitive_requires_exact_state_and_enum_types():
    v2 = _v2()

    with pytest.raises(TypeError, match="PrimitiveKindV2"):
        _primitive(v2, kind="wheel_motion")
    with pytest.raises(TypeError, match="PoseStateV2"):
        _primitive(v2, start_state=(0.0, 0.0, 0.0))
    with pytest.raises(TypeError, match="ValidationLevelV2"):
        _primitive(v2, validation_level="L2")


def test_typed_route_rejects_empty_mixed_and_disconnected_primitives():
    v2 = _v2()

    with pytest.raises(ValueError, match="nonempty"):
        v2.TypedRouteV2(v2.PlatformKindV2.WHEEL, (), 0.0)
    with pytest.raises(TypeError, match="PlatformKindV2"):
        v2.TypedRouteV2("wheel", (_primitive(v2),), 0.0)
    with pytest.raises(TypeError, match="tuple"):
        v2.TypedRouteV2(v2.PlatformKindV2.WHEEL, [_primitive(v2)], 0.0)

    mixed = (
        _primitive(v2, kind=v2.PrimitiveKindV2.WHEEL_MOTION),
        _primitive(v2, kind=v2.PrimitiveKindV2.LEG_STEP),
    )
    with pytest.raises(ValueError, match="platform"):
        _route(v2, primitives=mixed)

    first = _primitive(v2)
    disconnected = _primitive(
        v2,
        start_state=v2.PoseStateV2(9.0, 9.0, 0.0),
        end_state=v2.PoseStateV2(10.0, 9.0, 0.0),
    )
    with pytest.raises(ValueError, match="connected"):
        _route(v2, primitives=(first, disconnected))


def test_typed_route_rejects_nonfinite_cost_and_nonbool_completeness():
    v2 = _v2()

    with pytest.raises(ValueError, match="finite"):
        _route(v2, total_cost=float("nan"))
    with pytest.raises(TypeError, match="finite"):
        _route(v2, total_cost=True)
    with pytest.raises(TypeError, match="bool"):
        v2.TypedRouteV2(
            v2.PlatformKindV2.WHEEL,
            (_primitive(v2),),
            3.5,
            is_complete=1,
        )


def test_evidence_contracts_are_frozen_and_reject_mutable_containers():
    v2 = _v2()
    projection = _projection(v2)
    cost = _cost(v2)
    telemetry = _telemetry(v2)
    cache = _cache(v2)
    failure_evidence = _failure_evidence(v2)

    with pytest.raises(FrozenInstanceError):
        projection.source = "other"
    with pytest.raises(FrozenInstanceError):
        cost.total_cost = 9.0
    with pytest.raises(FrozenInstanceError):
        telemetry.elapsed_s = 9.0
    with pytest.raises(FrozenInstanceError):
        cache.hit = True
    with pytest.raises(FrozenInstanceError):
        failure_evidence.stage = "other"
    with pytest.raises(TypeError, match="tuple"):
        v2.ObservationProjectionV2("source", [v2.PoseStateV2(0.0, 0.0, 0.0)], 1.0, 1.0)
    with pytest.raises(TypeError, match="tuple"):
        v2.FailureEvidenceV2("search", ("failed",), {"expanded": 8})


def test_evidence_contracts_fail_closed_on_numeric_bool_and_string_inputs():
    v2 = _v2()

    with pytest.raises(TypeError, match="finite"):
        v2.ObservationProjectionV2("source", (), True, 0.0)
    with pytest.raises(ValueError, match="finite"):
        v2.CostBreakdownV2(1.0, 1.0, 1.0, float("inf"), 3.0)
    with pytest.raises(ValueError, match="components"):
        v2.CostBreakdownV2(1.0, 1.0, 1.0, 1.0, 5.0)
    with pytest.raises(TypeError, match="integer"):
        v2.SearchTelemetryV2(True, 1, 0, 0, 0, 0.1, False, False, False, "done")
    with pytest.raises(TypeError, match="bool"):
        v2.SearchTelemetryV2(1, 1, 0, 0, 0, 0.1, 0, False, False, "done")
    with pytest.raises(ValueError, match="termination_reason"):
        v2.SearchTelemetryV2(1, 1, 0, 0, 0, 0.1, False, False, False, " ")
    with pytest.raises(ValueError, match="cache_namespace"):
        v2.CacheEvidenceV2("", "key", False)
    with pytest.raises(TypeError, match="bool"):
        v2.CacheEvidenceV2("namespace", "key", 1)


def test_failure_evidence_requires_sorted_unique_scalar_details():
    v2 = _v2()

    with pytest.raises(ValueError, match="sorted"):
        v2.FailureEvidenceV2("search", (), (("z", 1), ("a", 2)))
    with pytest.raises(ValueError, match="unique"):
        v2.FailureEvidenceV2("search", (), (("a", 1), ("a", 2)))
    with pytest.raises(TypeError, match="scalar"):
        v2.FailureEvidenceV2("search", (), (("items", []),))
    with pytest.raises(ValueError, match="finite"):
        v2.FailureEvidenceV2("search", (), (("score", float("nan")),))
    with pytest.raises(ValueError, match="detail key"):
        v2.FailureEvidenceV2("search", (), (("", 1),))


def test_validation_evidence_requires_exact_level_and_immutable_checks():
    v2 = _v2()

    with pytest.raises(TypeError, match="ValidationLevelV2"):
        v2.ValidationEvidenceV2("validator", "L2", True, ())
    with pytest.raises(TypeError, match="tuple"):
        v2.ValidationEvidenceV2("validator", v2.ValidationLevelV2.L2, True, ["finite"])


def test_success_requires_complete_matching_l2_route_and_cost():
    v2 = _v2()

    with pytest.raises(ValueError, match="complete"):
        _success(v2, route=_route(v2, is_complete=False))
    with pytest.raises(ValueError, match="passed"):
        _success(v2, validation=_evidence(v2, passed=False))
    with pytest.raises(ValueError, match="L2"):
        _success(v2, validation=_evidence(v2, level=v2.ValidationLevelV2.L1))
    with pytest.raises(ValueError, match="platform"):
        _success(v2, platform_kind=v2.PlatformKindV2.LEGGED, route=_route(v2))
    with pytest.raises(ValueError, match="total_cost"):
        _success(v2, cost_breakdown=_cost(v2, total_cost=3.5001))


def test_full_success_and_failure_results_are_mutually_exclusive_and_frozen():
    v2 = _v2()
    success = _success(v2)
    failure = _failure(v2, platform_kind=v2.PlatformKindV2.WHEEL)
    unresolved_platform_failure = _failure(v2)

    assert success.schema_version == "path-planner-v2-planning/v1"
    assert failure.schema_version == "path-planner-v2-planning/v1"
    assert success.platform_kind is v2.PlatformKindV2.WHEEL
    assert failure.evidence.stage == "search"
    assert unresolved_platform_failure.platform_kind is None
    assert unresolved_platform_failure.evidence.stage == "profile_resolution"
    assert isinstance(success, v2.PlanningOutcomeV2)
    assert isinstance(failure, v2.PlanningOutcomeV2)
    assert not hasattr(success, "category")
    assert not hasattr(failure, "route")
    with pytest.raises(FrozenInstanceError):
        success.request_id = "other"

    with pytest.raises(ValueError, match="schema_version"):
        v2.PlanningSuccessV2(**{**{field.name: getattr(success, field.name) for field in fields(success)}, "schema_version": "other"})
    with pytest.raises(TypeError, match="FailureCategoryV2"):
        v2.PlanningFailureV2(
            request_id="request-001",
            platform_kind=None,
            category="timeout",
            reason_code="deadline",
            evidence=_failure_evidence(v2),
            search_telemetry=_telemetry(v2),
        )
    with pytest.raises(ValueError, match="reason_code"):
        v2.PlanningFailureV2(
            request_id="request-001",
            platform_kind=None,
            category=v2.FailureCategoryV2.TIMEOUT,
            reason_code=" ",
            evidence=_failure_evidence(v2),
            search_telemetry=_telemetry(v2),
        )
