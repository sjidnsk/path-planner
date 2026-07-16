from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep

import pytest

import path_planner.v2 as v2
from path_planner.v2.cache import (
    VALIDATION_CACHE_SCHEMA_VERSION_V2,
    ValidationCacheKeyV2,
    ValidationCacheV2,
)
from path_planner.v2.contracts import ValidationEvidenceV2, ValidationLevelV2


_PROFILE_HASH = "1" * 64
_TERRAIN_HASH = "2" * 64
_PRIMITIVE_HASH = "3" * 64
_OBJECTIVE_HASH = "4" * 64


def _key(**overrides: object) -> ValidationCacheKeyV2:
    values = {
        "schema_version": VALIDATION_CACHE_SCHEMA_VERSION_V2,
        "platform_profile_hash": _PROFILE_HASH,
        "terrain_snapshot_hash": _TERRAIN_HASH,
        "primitive_hash": _PRIMITIVE_HASH,
        "validation_level": ValidationLevelV2.L1,
        "objective_profile_hash": None,
    }
    values.update(overrides)
    return ValidationCacheKeyV2(**values)


def _evidence(
    *,
    level: ValidationLevelV2 = ValidationLevelV2.L1,
    passed: bool = True,
    check: str = "route_l1_valid",
) -> ValidationEvidenceV2:
    return ValidationEvidenceV2(
        validator_id="path-planner-v2-route-lazy-l1/v1",
        level=level,
        passed=passed,
        checks=(check,),
    )


def test_cache_contract_is_public_frozen_and_slotted() -> None:
    assert v2.VALIDATION_CACHE_SCHEMA_VERSION_V2 == VALIDATION_CACHE_SCHEMA_VERSION_V2
    assert v2.ValidationCacheKeyV2 is ValidationCacheKeyV2
    assert v2.ValidationCacheV2 is ValidationCacheV2

    key = _key()
    assert not hasattr(key, "__dict__")
    with pytest.raises(FrozenInstanceError):
        key.primitive_hash = "5" * 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"schema_version": ""},
        {"schema_version": "path-planner-v2-validation-cache/v999"},
        {"schema_version": type("Schema", (str,), {})(VALIDATION_CACHE_SCHEMA_VERSION_V2)},
        {"platform_profile_hash": "1" * 63},
        {"terrain_snapshot_hash": "A" * 64},
        {"primitive_hash": True},
        {"validation_level": "L1"},
        {"objective_profile_hash": ""},
    ],
)
def test_complete_key_rejects_missing_drift_malformed_and_subclass_values(kwargs) -> None:
    if not kwargs:
        with pytest.raises((TypeError, ValueError), match="complete cache key"):
            ValidationCacheKeyV2()
        return

    values = {
        "schema_version": VALIDATION_CACHE_SCHEMA_VERSION_V2,
        "platform_profile_hash": _PROFILE_HASH,
        "terrain_snapshot_hash": _TERRAIN_HASH,
        "primitive_hash": _PRIMITIVE_HASH,
        "validation_level": ValidationLevelV2.L1,
        "objective_profile_hash": None,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match="complete cache key"):
        ValidationCacheKeyV2(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "path-planner-v2-validation-cache/v2"),
        ("platform_profile_hash", "a" * 64),
        ("terrain_snapshot_hash", "b" * 64),
        ("primitive_hash", "c" * 64),
        ("validation_level", ValidationLevelV2.L0),
        ("objective_profile_hash", _OBJECTIVE_HASH),
    ],
)
def test_every_complete_key_field_changes_identity(field, value) -> None:
    cache = ValidationCacheV2()
    cache.put(_key(), _evidence())
    drifted = _key()
    if field == "schema_version":
        object.__setattr__(drifted, field, value)
    else:
        drifted = _key(**{field: value})

    assert cache.get(drifted) is None
    assert cache.entry_count == 1
    assert cache.hit_count == 0
    assert cache.miss_count == 1


def test_equal_distinct_complete_key_hits_by_canonical_value() -> None:
    cache = ValidationCacheV2()
    first = _key()
    second = _key()
    assert first is not second

    cache.put(first, _evidence())

    assert cache.get(second) == _evidence()
    assert (cache.hit_count, cache.miss_count, cache.entry_count) == (1, 0, 1)


def test_forged_key_read_fails_closed_as_miss_and_write_is_rejected() -> None:
    cache = ValidationCacheV2()
    key = _key()
    cache.put(key, _evidence())
    object.__setattr__(key, "primitive_hash", "forged")

    assert cache.get(key) is None
    assert cache.miss_count == 1
    with pytest.raises((TypeError, ValueError), match="complete cache key"):
        cache.put(key, _evidence())


def test_put_rejects_level_mismatch_wrong_types_and_transient_evidence() -> None:
    cache = ValidationCacheV2()

    with pytest.raises(ValueError, match="evidence level"):
        cache.put(_key(), _evidence(level=ValidationLevelV2.L0))
    with pytest.raises(TypeError, match="exact ValidationEvidenceV2"):
        cache.put(_key(), object())
    with pytest.raises(ValueError, match="not cacheable"):
        cache.put(
            _key(),
            _evidence(passed=False, check="planning_deadline_expired"),
        )
    with pytest.raises(ValueError, match="not cacheable"):
        cache.put(
            _key(),
            _evidence(passed=False, check="lazy_validation_internal_contract_mismatch"),
        )


def test_identical_replay_is_idempotent_and_conflict_is_rejected() -> None:
    cache = ValidationCacheV2()
    key = _key()
    evidence = _evidence()

    cache.put(key, evidence)
    cache.put(key, _evidence())
    assert cache.entry_count == 1

    with pytest.raises(ValueError, match="conflicting validation evidence"):
        cache.put(
            key,
            _evidence(passed=False, check="terrain_unknown"),
        )
    assert cache.entry_count == 1


def test_concurrent_conflicting_put_has_one_winner_and_one_rejection() -> None:
    class RacingDict(dict):
        def get(self, key, default=None):
            result = super().get(key, default)
            sleep(0.05)
            return result

    cache = ValidationCacheV2()
    cache._entries = RacingDict()
    assert hasattr(cache, "_lock")
    key = _key()
    passed = _evidence()
    rejected = _evidence(passed=False, check="terrain_unknown")
    barrier = Barrier(2)

    def write(evidence):
        barrier.wait()
        try:
            cache.put(key, evidence)
        except ValueError:
            return ("rejected", evidence)
        return ("stored", evidence)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(write, (passed, rejected)))

    winners = tuple(evidence for status, evidence in outcomes if status == "stored")
    failures = tuple(evidence for status, evidence in outcomes if status == "rejected")
    assert len(winners) == 1
    assert len(failures) == 1
    assert cache.entry_count == 1
    assert cache.get(key) == winners[0]


def test_cache_defensively_copies_evidence_on_write_and_read() -> None:
    cache = ValidationCacheV2()
    key = _key()
    source = _evidence()
    cache.put(key, source)
    object.__setattr__(source, "checks", ("forged_source",))

    first = cache.get(key)
    assert first == _evidence()
    assert first is not source
    object.__setattr__(first, "checks", ("forged_read",))

    second = cache.get(key)
    assert second == _evidence()
    assert second is not first


def test_read_of_wrong_key_type_is_a_deterministic_miss() -> None:
    cache = ValidationCacheV2()

    assert cache.get(object()) is None
    assert (cache.hit_count, cache.miss_count, cache.entry_count) == (0, 1, 0)
