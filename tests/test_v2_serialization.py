from importlib import import_module

import pytest


def _v2():
    return import_module("path_planner.v2")


def _success(v2):
    return v2.PlanningSuccessV2(
        request_id="request-月",
        route=v2.TypedRouteV2(
            platform_kind=v2.PlatformKindV2.WHEEL,
            states=(
                v2.PoseStateV2(0.0, 0.0, 0.0),
                v2.PoseStateV2(1.0, 2.0, 0.5),
            ),
            total_cost=3.5,
        ),
        validation_evidence=v2.ValidationEvidenceV2(
            validator_id="route-safety/v1",
            passed=True,
            checks=("finite", "collision-free"),
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
    assert encoded.startswith(b'{"request_id":"request-')


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
