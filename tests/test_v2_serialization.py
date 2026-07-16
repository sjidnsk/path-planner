import json
from importlib import import_module

import pytest


def _v2():
    return import_module("path_planner.v2")


def _success(v2):
    start = v2.PoseStateV2(0.0, 0.0, 0.0)
    end = v2.PoseStateV2(1.0, 2.0, 0.5)
    return v2.PlanningSuccessV2(
        request_id="request-月",
        platform_kind=v2.PlatformKindV2.WHEEL,
        route=v2.TypedRouteV2(
            platform_kind=v2.PlatformKindV2.WHEEL,
            primitives=(
                v2.RoutePrimitiveV2(
                    kind=v2.PrimitiveKindV2.WHEEL_MOTION,
                    start_state=start,
                    end_state=end,
                    duration_s=1.25,
                    distance_m=2.5,
                    energy_cost=0.5,
                    observation_contribution=4.0,
                    validation_level=v2.ValidationLevelV2.L2,
                ),
            ),
            total_cost=3.5,
        ),
        observation_projection=v2.ObservationProjectionV2(
            source="terrain-observation/v1",
            sample_states=(start, end),
            expected_new_observed_cells=4.0,
            expected_information_gain=0.75,
        ),
        cost_breakdown=v2.CostBreakdownV2(1.0, 0.5, 0.75, 1.25, 3.5),
        validation_evidence=v2.ValidationEvidenceV2(
            validator_id="route-safety/v1",
            level=v2.ValidationLevelV2.L2,
            passed=True,
            checks=("finite", "collision-free"),
        ),
        search_telemetry=v2.SearchTelemetryV2(
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
        ),
        cache_evidence=v2.CacheEvidenceV2("planner-v2", "request-月", False),
    )


def _failure(v2):
    return v2.PlanningFailureV2(
        request_id="request-failed",
        platform_kind=None,
        category=v2.FailureCategoryV2.NO_COMPLETE_ROUTE,
        reason_code="search_exhausted",
        evidence=v2.FailureEvidenceV2(
            stage="profile_resolution",
            checks=("platform-kind-unresolved",),
            details=(("profile_id", "unknown"),),
        ),
        search_telemetry=v2.SearchTelemetryV2(
            expanded_states=8,
            generated_primitives=3,
            rejected_l0=1,
            rejected_l1=1,
            rejected_l2=1,
            elapsed_s=0.25,
            timed_out=False,
            accelerator_used=False,
            ackermann_feasible_claimed=False,
            termination_reason="frontier_exhausted",
        ),
    )


def test_canonical_json_bytes_are_stable_utf8_and_compact():
    v2 = _v2()
    success = _success(v2)

    encoded = v2.canonical_json_bytes(success)

    assert encoded == v2.canonical_json_bytes(success)
    assert b"\n" not in encoded
    assert b" " not in encoded
    assert "月".encode("utf-8") in encoded
    assert encoded.startswith(b'{"cache_evidence":')


def test_canonical_json_serializes_every_success_result_field():
    v2 = _v2()

    payload = json.loads(v2.canonical_json_bytes(_success(v2)))

    assert set(payload) == {
        "schema_version",
        "request_id",
        "platform_kind",
        "route",
        "observation_projection",
        "cost_breakdown",
        "validation_evidence",
        "search_telemetry",
        "cache_evidence",
    }
    assert payload["platform_kind"] == "wheel"
    assert payload["route"] == {
        "is_complete": True,
        "platform_kind": "wheel",
        "primitives": [
            {
                "distance_m": 2.5,
                "duration_s": 1.25,
                "end_state": {"heading_rad": 0.5, "x_m": 1.0, "y_m": 2.0},
                "energy_cost": 0.5,
                "kind": "wheel_motion",
                "observation_contribution": 4.0,
                "start_state": {"heading_rad": 0.0, "x_m": 0.0, "y_m": 0.0},
                "validation_level": "L2",
            }
        ],
        "total_cost": 3.5,
    }
    assert payload["observation_projection"]["source"] == "terrain-observation/v1"
    assert payload["cost_breakdown"] == {
        "distance_cost": 1.0,
        "energy_cost": 0.75,
        "risk_cost": 0.5,
        "time_cost": 1.25,
        "total_cost": 3.5,
    }
    assert payload["validation_evidence"]["level"] == "L2"
    assert payload["search_telemetry"]["ackermann_feasible_claimed"] is False
    assert payload["cache_evidence"] == {
        "cache_key": "request-月",
        "cache_namespace": "planner-v2",
        "hit": False,
    }


def test_canonical_json_serializes_every_failure_result_field():
    v2 = _v2()

    payload = json.loads(v2.canonical_json_bytes(_failure(v2)))

    assert set(payload) == {
        "schema_version",
        "request_id",
        "platform_kind",
        "category",
        "reason_code",
        "evidence",
        "search_telemetry",
    }
    assert payload["platform_kind"] is None
    assert payload["category"] == "no_complete_route"
    assert payload["evidence"] == {
        "checks": ["platform-kind-unresolved"],
        "details": [["profile_id", "unknown"]],
        "stage": "profile_resolution",
    }
    assert payload["search_telemetry"]["termination_reason"] == "frontier_exhausted"


def test_canonical_json_is_insensitive_to_mapping_insertion_order():
    v2 = _v2()
    left = {"z": 1, "a": {"second": 2, "first": 1}}
    right = {"a": {"first": 1, "second": 2}, "z": 1}

    assert v2.canonical_json_bytes(left) == v2.canonical_json_bytes(right)
    assert v2.canonical_json_bytes(left) == b'{"a":{"first":1,"second":2},"z":1}'


def test_canonical_json_preserves_tuple_order():
    v2 = _v2()

    assert v2.canonical_json_bytes(("first", "second")) != v2.canonical_json_bytes(
        ("second", "first")
    )
    assert v2.canonical_json_bytes(("first", "second")) == b'["first","second"]'


def test_canonical_json_recursively_supports_enums_lists_and_scalars():
    v2 = _v2()
    value = {
        "kind": v2.PlatformKindV2.LEGGED,
        "items": [None, True, 3, 1.5, "text"],
    }

    assert v2.canonical_json_bytes(value) == (
        b'{"items":[null,true,3,1.5,"text"],"kind":"legged"}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_values(value):
    v2 = _v2()

    with pytest.raises(ValueError, match="finite"):
        v2.canonical_json_bytes({"nested": [value]})


def test_canonical_json_rejects_non_string_mapping_keys():
    v2 = _v2()

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        v2.canonical_json_bytes({1: "value"})


def test_canonical_json_rejects_unknown_objects():
    v2 = _v2()

    with pytest.raises(TypeError, match="unsupported type"):
        v2.canonical_json_bytes(object())
