from dataclasses import FrozenInstanceError
from importlib import import_module

import pytest


def _v2():
    return import_module("path_planner.v2")


def _route(v2, *, is_complete: bool = True):
    return v2.TypedRouteV2(
        platform_kind=v2.PlatformKindV2.WHEEL,
        states=(
            v2.PoseStateV2(0.0, 0.0, 0.0),
            v2.PoseStateV2(1.0, 2.0, 0.5),
        ),
        total_cost=3.5,
        is_complete=is_complete,
    )


def _evidence(v2, *, passed: bool = True):
    return v2.ValidationEvidenceV2(
        validator_id="route-safety/v1",
        passed=passed,
        checks=("finite", "collision-free"),
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


def test_typed_route_requires_typed_nonempty_finite_state_sequence():
    v2 = _v2()

    with pytest.raises(ValueError, match="nonempty"):
        v2.TypedRouteV2(v2.PlatformKindV2.WHEEL, (), 0.0)
    with pytest.raises(TypeError, match="PlatformKindV2"):
        v2.TypedRouteV2("wheel", (v2.PoseStateV2(0.0, 0.0, 0.0),), 0.0)
    with pytest.raises(TypeError, match="tuple"):
        v2.TypedRouteV2(v2.PlatformKindV2.WHEEL, [v2.PoseStateV2(0.0, 0.0, 0.0)], 0.0)
    with pytest.raises(ValueError, match="finite"):
        v2.TypedRouteV2(
            v2.PlatformKindV2.WHEEL,
            (v2.PoseStateV2(0.0, 0.0, 0.0),),
            float("nan"),
        )


def test_success_requires_a_complete_route_and_passing_validation():
    v2 = _v2()

    with pytest.raises(ValueError, match="complete"):
        v2.PlanningSuccessV2("request-001", _route(v2, is_complete=False), _evidence(v2))
    with pytest.raises(ValueError, match="passed"):
        v2.PlanningSuccessV2("request-001", _route(v2), _evidence(v2, passed=False))


def test_success_and_failure_are_mutually_exclusive_outcomes_with_fixed_schema():
    v2 = _v2()
    success = v2.PlanningSuccessV2("request-001", _route(v2), _evidence(v2))
    failure = v2.PlanningFailureV2(
        "request-001",
        v2.FailureCategoryV2.NO_COMPLETE_ROUTE,
        "search_exhausted",
    )

    assert success.schema_version == "path-planner-v2-planning/v1"
    assert failure.schema_version == "path-planner-v2-planning/v1"
    assert isinstance(success, v2.PlanningOutcomeV2)
    assert isinstance(failure, v2.PlanningOutcomeV2)
    assert not hasattr(success, "failure_category")
    assert not hasattr(failure, "route")

    with pytest.raises(ValueError, match="schema_version"):
        v2.PlanningSuccessV2(
            "request-001",
            _route(v2),
            _evidence(v2),
            schema_version="other",
        )
    with pytest.raises(TypeError, match="FailureCategoryV2"):
        v2.PlanningFailureV2("request-001", "timeout", "deadline")
    with pytest.raises(ValueError, match="reason_code"):
        v2.PlanningFailureV2("request-001", v2.FailureCategoryV2.TIMEOUT, " ")
