from dataclasses import FrozenInstanceError, fields
from importlib import import_module
from inspect import Parameter, signature
from math import ceil, pi

import pytest

import path_planner.v2 as v2_package
import path_planner.v2.ballistics as ballistics_module
import path_planner.v2.oracles as oracles_package
import path_planner.v2.providers as providers_package
from path_planner.core.models import WorldPoint
from path_planner.v2.ballistics import BallisticStartV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineGridGeometryV2


_RESOURCE_FIELD_NAMES = (
    "ballistic_helper",
    "ballistic_helper_id",
    "landing_helper",
    "landing_helper_id",
    "memory_accounting_id",
    "max_ballistic_samples",
    "max_landing_candidates",
    "max_replay_steps",
    "max_exact_integer_bits",
    "max_exact_live_integer_slots",
    "hard_accounted_memory_bytes",
    "search_base_bytes",
    "search_record_bytes",
    "ballistic_build_base_bytes",
    "ballistic_build_per_sample_bytes",
    "ballistic_retained_per_sample_bytes",
    "landing_build_base_bytes",
    "landing_build_per_candidate_bytes",
    "exact_base_bytes",
    "exact_integer_slot_overhead_bytes",
    "exact_distinct_cell_bytes",
    "route_build_base_bytes",
    "route_primitive_bytes",
    "schema_version",
)

_NUMERIC_FIELD_VALUES = (
    ("max_ballistic_samples", 100_000),
    ("max_landing_candidates", 1_000_000),
    ("max_replay_steps", 100_000),
    ("max_exact_integer_bits", 262_144),
    ("max_exact_live_integer_slots", 128),
    ("hard_accounted_memory_bytes", 536_870_912),
    ("search_base_bytes", 4_096),
    ("search_record_bytes", 1_024),
    ("ballistic_build_base_bytes", 4_096),
    ("ballistic_build_per_sample_bytes", 96),
    ("ballistic_retained_per_sample_bytes", 48),
    ("landing_build_base_bytes", 65_536),
    ("landing_build_per_candidate_bytes", 384),
    ("exact_base_bytes", 4_096),
    ("exact_integer_slot_overhead_bytes", 16),
    ("exact_distinct_cell_bytes", 320),
    ("route_build_base_bytes", 4_096),
    ("route_primitive_bytes", 512),
)


class _AuthorityIntSubclass(int):
    pass


class _AuthorityIntCoercible:
    def __int__(self) -> int:
        return 1


class _AuthorityStrSubclass(str):
    pass


class _EqualityForgingCallable:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return ()

    def __eq__(self, _other: object) -> bool:
        return True


def _authority_module():
    return import_module("path_planner.v2.hopper_authority")


def _authority_values(authority) -> dict[str, object]:
    return {name: getattr(authority, name) for name in _RESOURCE_FIELD_NAMES}


def _fresh_authority(module):
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    return module.HopperResourceAuthorityV2(**_authority_values(authority))


def test_hopper_resource_authority_freezes_exact_record_singleton_and_values() -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2

    assert type(authority) is module.HopperResourceAuthorityV2
    assert tuple(field.name for field in fields(module.HopperResourceAuthorityV2)) == (
        _RESOURCE_FIELD_NAMES
    )
    assert not hasattr(authority, "__dict__")
    with pytest.raises(FrozenInstanceError):
        authority.max_replay_steps = 1

    assert authority.ballistic_helper is getattr(
        ballistics_module, "sample_ballistic_arc_capped_v2"
    )
    assert authority.ballistic_helper_id == "sample_ballistic_arc_capped/v1"
    assert authority.landing_helper is getattr(
        ballistics_module, "landing_zone_cells_capped_v2"
    )
    assert authority.landing_helper_id == "landing_zone_cells_capped/v1"
    assert authority.memory_accounting_id == "hopper_deterministic_admission_bytes/v1"
    assert authority.schema_version == "hopper-resource-authority/v1"
    assert tuple(
        (name, getattr(authority, name)) for name, _value in _NUMERIC_FIELD_VALUES
    ) == _NUMERIC_FIELD_VALUES

    assert authority.ballistic_build_base_bytes + (
        authority.ballistic_build_per_sample_bytes
        * authority.max_ballistic_samples
    ) == 9_604_096
    assert module.HOPPER_BALLISTIC_BUILD_MAX_BYTES_V2 == 9_604_096
    assert module.HOPPER_RETURNED_BALLISTIC_SAMPLE_BYTES_V2 == 48
    assert authority.landing_build_base_bytes + (
        authority.landing_build_per_candidate_bytes
        * authority.max_landing_candidates
    ) == 384_065_536
    assert module.HOPPER_LANDING_BUILD_MAX_BYTES_V2 == 384_065_536
    exact_slot_bytes = authority.exact_integer_slot_overhead_bytes + ceil(
        authority.max_exact_integer_bits / 8
    )
    assert exact_slot_bytes == 32_784
    assert module.HOPPER_EXACT_MAX_INTEGER_BITS_V2 == 262_144
    assert module.HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2 == 128
    assert module.HOPPER_EXACT_INTEGER_SLOT_BYTES_V2 == 32_784
    assert authority.exact_base_bytes + (
        authority.max_exact_live_integer_slots * exact_slot_bytes
    ) + (
        authority.max_replay_steps * authority.exact_distinct_cell_bytes
    ) == 36_200_448
    assert module.HOPPER_EXACT_ORACLE_MAX_BYTES_V2 == 36_200_448
    assert module.HOPPER_HARD_ACCOUNTED_MEMORY_BYTES_V2 == 536_870_912
    assert module.HOPPER_EXACT_DISTINCT_CELL_BYTES_V2 == 320
    sample_count = 7
    assert (
        authority.ballistic_retained_per_sample_bytes * sample_count
        + module.HOPPER_EXACT_ORACLE_MAX_BYTES_V2
        == 36_200_784
    )
    assert authority.search_base_bytes + 7 * authority.search_record_bytes == 11_264
    assert authority.route_build_base_bytes + 7 * authority.route_primitive_bytes == 7_680


def test_hopper_resource_authority_rejects_non_exact_record_field_types() -> None:
    module = _authority_module()
    values = _authority_values(module.HOPPER_RESOURCE_AUTHORITY_V2)

    for name, original in _NUMERIC_FIELD_VALUES:
        for bad_value in (
            True,
            _AuthorityIntSubclass(original),
            _AuthorityIntCoercible(),
        ):
            invalid = dict(values)
            invalid[name] = bad_value
            with pytest.raises(TypeError, match=rf"{name}.*exact int"):
                module.HopperResourceAuthorityV2(**invalid)
        for bad_value in (0, -1):
            invalid = dict(values)
            invalid[name] = bad_value
            with pytest.raises(ValueError, match=rf"{name}.*positive"):
                module.HopperResourceAuthorityV2(**invalid)
        invalid = dict(values)
        invalid[name] = original + 1
        with pytest.raises(ValueError, match=name):
            module.HopperResourceAuthorityV2(**invalid)

    for name in (
        "ballistic_helper_id",
        "landing_helper_id",
        "memory_accounting_id",
        "schema_version",
    ):
        invalid = dict(values)
        invalid[name] = _AuthorityStrSubclass(str(values[name]))
        with pytest.raises(TypeError, match=rf"{name}.*exact str"):
            module.HopperResourceAuthorityV2(**invalid)
        invalid = dict(values)
        invalid[name] = str(values[name]) + "-drift"
        with pytest.raises(ValueError, match=name):
            module.HopperResourceAuthorityV2(**invalid)

    invalid = dict(values)
    invalid["ballistic_helper"] = lambda: None
    with pytest.raises(ValueError, match="ballistic_helper.*trusted"):
        module.HopperResourceAuthorityV2(**invalid)
    invalid = dict(values)
    invalid["landing_helper"] = lambda: None
    with pytest.raises(ValueError, match="landing_helper.*trusted"):
        module.HopperResourceAuthorityV2(**invalid)


def test_hopper_resource_authority_tokens_follow_exact_frozen_order() -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    expected_in_memory = tuple(
        getattr(authority, name) for name in _RESOURCE_FIELD_NAMES
    )
    expected_lineage = (
        "hopper-resource-authority-lineage/v1",
        authority.ballistic_helper_id,
        authority.landing_helper_id,
        authority.memory_accounting_id,
        *(
            getattr(authority, name)
            for name, _value in _NUMERIC_FIELD_VALUES
        ),
        authority.schema_version,
    )

    assert type(module.HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2) is tuple
    assert module.HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2 == expected_in_memory
    assert module.HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2[0] is (
        authority.ballistic_helper
    )
    assert module.HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2[2] is (
        authority.landing_helper
    )
    assert type(module.HOPPER_RESOURCE_AUTHORITY_LINEAGE_TOKEN_V2) is tuple
    assert module.HOPPER_RESOURCE_AUTHORITY_LINEAGE_TOKEN_V2 == expected_lineage
    first_in_memory = module._hopper_resource_authority_in_memory_token_v2(
        authority
    )
    second_in_memory = module._hopper_resource_authority_in_memory_token_v2(
        authority
    )
    first_lineage = module._hopper_resource_authority_lineage_token_v2(authority)
    second_lineage = module._hopper_resource_authority_lineage_token_v2(authority)
    assert type(first_in_memory) is type(second_in_memory) is tuple
    assert type(first_lineage) is type(second_lineage) is tuple
    assert first_in_memory == second_in_memory == expected_in_memory
    assert first_lineage == second_lineage == expected_lineage


def test_captured_resource_helpers_use_sealed_caps_not_legacy_globals(
    monkeypatch,
) -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    ballistic_args = (
        BallisticStartV2(0.0, 0.0, 0.5),
        3.0,
        pi / 4.0,
        0.0,
        1.62,
        0.25,
    )
    landing_args = (
        WorldPoint(0.25, 0.25),
        0.05,
        0.99,
        FineGridGeometryV2(4, 4),
    )
    expected_ballistic = authority.ballistic_helper(
        *ballistic_args, max_sample_count=authority.max_ballistic_samples
    )
    expected_landing = authority.landing_helper(
        *landing_args, max_candidate_count=authority.max_landing_candidates
    )

    ballistic_cap_sentinel = object()
    landing_cap_sentinel = object()
    original_interior = ballistics_module._interior_sample_times
    original_perimeter = ballistics_module._iter_square_perimeter_cells
    legacy_wrapper_calls: list[str] = []
    materialization_phases: list[str] = []

    def auditing_interior(*args, **kwargs):
        assert ballistics_module.MAX_BALLISTIC_SAMPLES_V2 is ballistic_cap_sentinel
        materialization_phases.append("ballistic")
        return original_interior(*args, **kwargs)

    def auditing_perimeter(*args, **kwargs):
        assert (
            ballistics_module.MAX_LANDING_ZONE_CANDIDATES_V2
            is landing_cap_sentinel
        )
        materialization_phases.append("landing")
        yield from original_perimeter(*args, **kwargs)

    def forbidden_legacy_ballistic(*_args, **_kwargs):
        legacy_wrapper_calls.append("ballistic")
        raise AssertionError("captured path must not call the legacy ballistic wrapper")

    def forbidden_legacy_landing(*_args, **_kwargs):
        legacy_wrapper_calls.append("landing")
        raise AssertionError("captured path must not call the legacy landing wrapper")

    monkeypatch.setattr(
        ballistics_module, "MAX_BALLISTIC_SAMPLES_V2", ballistic_cap_sentinel
    )
    monkeypatch.setattr(
        ballistics_module,
        "MAX_LANDING_ZONE_CANDIDATES_V2",
        landing_cap_sentinel,
    )
    monkeypatch.setattr(ballistics_module, "_interior_sample_times", auditing_interior)
    monkeypatch.setattr(
        ballistics_module, "_iter_square_perimeter_cells", auditing_perimeter
    )
    monkeypatch.setattr(
        ballistics_module, "sample_ballistic_arc", forbidden_legacy_ballistic
    )
    monkeypatch.setattr(
        ballistics_module, "landing_zone_cells", forbidden_legacy_landing
    )
    actual_ballistic = module._call_captured_ballistic_helper_v2(
        authority, *ballistic_args
    )
    actual_landing = module._call_captured_landing_helper_v2(authority, *landing_args)
    assert canonical_json_bytes(actual_ballistic) == canonical_json_bytes(
        expected_ballistic
    )
    assert canonical_json_bytes(actual_landing) == canonical_json_bytes(
        expected_landing
    )
    assert "ballistic" in materialization_phases
    assert "landing" in materialization_phases
    assert legacy_wrapper_calls == []


def test_captured_resource_helpers_fail_before_rebound_module_callable(
    monkeypatch,
) -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    calls: list[str] = []

    def rebound_ballistic(*_args, **_kwargs):
        calls.append("ballistic")
        return ()

    def rebound_landing(*_args, **_kwargs):
        calls.append("landing")
        return ()

    monkeypatch.setattr(
        ballistics_module, "sample_ballistic_arc_capped_v2", rebound_ballistic
    )
    with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
        module._call_captured_ballistic_helper_v2(
            authority,
            object(),
            3.0,
            pi / 4.0,
            0.0,
            1.62,
            0.25,
        )
    assert calls == []

    monkeypatch.undo()
    monkeypatch.setattr(
        ballistics_module, "landing_zone_cells_capped_v2", rebound_landing
    )
    with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
        module._call_captured_landing_helper_v2(
            authority,
            object(),
            0.05,
            0.99,
            FineGridGeometryV2(4, 4),
        )
    assert calls == []


def test_captured_resource_helpers_fail_before_every_token_field_drift() -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    canonical_clone = _fresh_authority(module)
    assert canonical_clone == authority
    assert canonical_clone is not authority

    authority_subclass = type(
        "HopperResourceAuthoritySubclassV2",
        (module.HopperResourceAuthorityV2,),
        {"__slots__": ()},
    )
    forged_subclass = object.__new__(authority_subclass)
    for field_name in _RESOURCE_FIELD_NAMES:
        object.__setattr__(
            forged_subclass,
            field_name,
            getattr(authority, field_name),
        )

    invalid_ballistic_args = (
        object(),
        3.0,
        pi / 4.0,
        0.0,
        1.62,
        0.25,
    )
    invalid_landing_args = (
        object(),
        0.05,
        0.99,
        FineGridGeometryV2(4, 4),
    )

    for forged in (canonical_clone, forged_subclass):
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            module._call_captured_ballistic_helper_v2(forged, *invalid_ballistic_args)
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            module._call_captured_landing_helper_v2(forged, *invalid_landing_args)

    equality_forgers: list[_EqualityForgingCallable] = []
    for field_name in _RESOURCE_FIELD_NAMES:
        original = getattr(authority, field_name)
        if field_name in ("ballistic_helper", "landing_helper"):
            drifted_values = (_EqualityForgingCallable(),)
            equality_forgers.extend(drifted_values)
        elif field_name in {
            "ballistic_helper_id",
            "landing_helper_id",
            "memory_accounting_id",
            "schema_version",
        }:
            drifted_values = (
                original + "-drift",
                _AuthorityStrSubclass(original),
            )
        else:
            drifted_values = (
                original + 1,
                _AuthorityIntSubclass(original),
            )
        for drifted in drifted_values:
            object.__setattr__(authority, field_name, drifted)
            try:
                with pytest.raises(
                    ValueError, match="^hopper_authority_contract_mismatch$"
                ):
                    module._call_captured_ballistic_helper_v2(
                        authority, *invalid_ballistic_args
                    )
                with pytest.raises(
                    ValueError, match="^hopper_authority_contract_mismatch$"
                ):
                    module._call_captured_landing_helper_v2(
                        authority, *invalid_landing_args
                    )
            finally:
                object.__setattr__(authority, field_name, original)

    assert all(forged.calls == 0 for forged in equality_forgers)


def test_exact_integer_arena_admits_bit_ceiling_before_operation_and_unwinds() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    observations: list[tuple[int, int]] = []

    def operation() -> int:
        observations.append((arena.live_integer_slots, arena.peak_live_integer_slots))
        return 7

    assert arena.run_admitted(
        result_bit_bound=262_144,
        operation=operation,
    ) == 7
    assert observations == [(1, 1)]
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 1


def test_exact_integer_arena_rejects_invalid_or_overlimit_bound_before_operation() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    operation_calls = 0

    def forbidden_operation() -> int:
        nonlocal operation_calls
        operation_calls += 1
        return 1

    for invalid in (
        True,
        _AuthorityIntSubclass(1),
        _AuthorityIntCoercible(),
        0,
        -1,
        262_145,
    ):
        with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
            arena.run_admitted(
                result_bit_bound=invalid,
                operation=forbidden_operation,
            )
    assert operation_calls == 0
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 0


def test_exact_integer_arena_rejects_slot_129_before_operation_and_recovers() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    forbidden_calls = 0

    def forbidden_operation() -> int:
        nonlocal forbidden_calls
        forbidden_calls += 1
        return 1

    def descend(remaining: int) -> int:
        if remaining == 0:
            return arena.run_admitted(
                result_bit_bound=1,
                operation=forbidden_operation,
            )
        return arena.run_admitted(
            result_bit_bound=1,
            operation=lambda: descend(remaining - 1),
        )

    with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
        descend(128)
    assert forbidden_calls == 0
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 128
    assert arena.run_admitted(result_bit_bound=1, operation=lambda: 1) == 1
    assert arena.live_integer_slots == 0


def test_exact_integer_arena_requires_live_canonical_resource_authority() -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    clone = _fresh_authority(module)
    authority_subclass = type(
        "HopperResourceAuthorityArenaSubclassV2",
        (module.HopperResourceAuthorityV2,),
        {"__slots__": ()},
    )
    forged_subclass = object.__new__(authority_subclass)
    for field_name in _RESOURCE_FIELD_NAMES:
        object.__setattr__(
            forged_subclass,
            field_name,
            getattr(authority, field_name),
        )

    for forged in (clone, forged_subclass):
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            module._HopperExactIntegerArenaV2(forged)

    operation_calls = 0

    def forbidden_operation() -> int:
        nonlocal operation_calls
        operation_calls += 1
        return 1

    for field_name, drifted in (
        ("max_exact_integer_bits", 262_145),
        ("max_exact_integer_bits", _AuthorityIntSubclass(262_144)),
        ("max_exact_live_integer_slots", 129),
        ("max_exact_live_integer_slots", _AuthorityIntSubclass(128)),
        ("exact_integer_slot_overhead_bytes", 17),
        ("exact_integer_slot_overhead_bytes", _AuthorityIntSubclass(16)),
    ):
        original = getattr(authority, field_name)
        object.__setattr__(authority, field_name, drifted)
        try:
            with pytest.raises(
                ValueError, match="^hopper_authority_contract_mismatch$"
            ):
                module._HopperExactIntegerArenaV2(authority)
        finally:
            object.__setattr__(authority, field_name, original)

        arena = module._HopperExactIntegerArenaV2(authority)
        object.__setattr__(authority, field_name, drifted)
        try:
            with pytest.raises(
                ValueError, match="^hopper_authority_contract_mismatch$"
            ):
                arena.run_admitted(
                    result_bit_bound=1,
                    operation=forbidden_operation,
                )
        finally:
            object.__setattr__(authority, field_name, original)

    assert operation_calls == 0


def test_neutral_resource_authority_adds_no_fixture_registry_or_package_export() -> None:
    module = _authority_module()
    for forbidden in (
        "HopperProviderAuthorityV2",
        "HopperParameterSetRecordV2",
        "HOPPER_PARAMETER_SET_REGISTRY_V2",
        "HOPPER_GATE5B_ALGORITHM_FIXTURE_V1",
        "hopper_gate5b_algorithm_fixture_v1",
    ):
        assert not hasattr(module, forbidden)

    for package in (v2_package, oracles_package, providers_package):
        exports = tuple(getattr(package, "__all__", ()))
        assert "HopperResourceAuthorityV2" not in exports
        assert "HOPPER_RESOURCE_AUTHORITY_V2" not in exports
        assert not hasattr(package, "HopperResourceAuthorityV2")
        assert not hasattr(package, "HOPPER_RESOURCE_AUTHORITY_V2")


# This CPS entry point is only the atomic N-slot reservation, exact result-tuple
# bit audit, and consumer-lifetime substrate for 11B2's operation-specific exact
# kernel.  It does not authorize arbitrary callback arithmetic or itself prove
# nonescape.  11B2 must use bound-deriving wrappers plus an AST gate for direct
# generic/legacy/Fraction/escape paths; the sole Cell promotion is separately
# frozen and transferred into the distinct-cell ledger.
def test_exact_integer_cps_reserves_all_slots_through_consumer_and_releases() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    method = module._HopperExactIntegerArenaV2.consume_admitted_integers
    parameters = tuple(signature(method).parameters.values())
    assert tuple(parameter.name for parameter in parameters) == (
        "self",
        "result_bit_bounds",
        "operation",
        "consumer",
    )
    assert parameters[0].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert all(parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters[1:])
    assert all(parameter.default is Parameter.empty for parameter in parameters)

    observations: list[tuple[str, int, int, bool | None, str | None]] = []
    produced = (0, -7, 255)
    semantic_result = {"opaque": [object()]}

    def operation() -> tuple[int, ...]:
        observations.append(
            (
                "operation",
                arena.live_integer_slots,
                arena.peak_live_integer_slots,
                None,
                None,
            )
        )
        return produced

    def consumer(values: tuple[int, ...]):
        observations.append(
            (
                "consumer",
                arena.live_integer_slots,
                arena.peak_live_integer_slots,
                values is produced,
                ",".join(str(value.bit_length()) for value in values),
            )
        )
        assert values is produced
        return semantic_result

    assert arena.consume_admitted_integers(
        result_bit_bounds=(1, 3, 262_144),
        operation=operation,
        consumer=consumer,
    ) is semantic_result
    assert observations == [
        ("operation", 3, 3, None, None),
        ("consumer", 3, 3, True, "0,3,8"),
    ]
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 3

    max_result_holder: list[tuple[int, ...]] = []

    def max_bit_operation() -> tuple[int, ...]:
        assert arena.live_integer_slots == 1
        result = (1 << 262_143,)
        assert result[0].bit_length() == 262_144
        max_result_holder.append(result)
        return result

    def max_bit_consumer(values: tuple[int, ...]) -> str:
        assert arena.live_integer_slots == 1
        assert values is max_result_holder[0]
        assert values[0].bit_length() == 262_144
        max_result_holder.clear()
        return "max-bit-result-consumed"

    assert arena.consume_admitted_integers(
        result_bit_bounds=(262_144,),
        operation=max_bit_operation,
        consumer=max_bit_consumer,
    ) == "max-bit-result-consumed"
    assert max_result_holder == []
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 3


def test_exact_integer_cps_rejects_invalid_bounds_count_and_callables_preoperation() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    operation_calls = 0
    consumer_calls = 0

    class _BoundsTupleSubclass(tuple):
        pass

    def operation() -> tuple[int, ...]:
        nonlocal operation_calls
        operation_calls += 1
        return (1,)

    def consumer(_values: tuple[int, ...]) -> object:
        nonlocal consumer_calls
        consumer_calls += 1
        return object()

    invalid_bounds = (
        [],
        _BoundsTupleSubclass((1,)),
        (),
        (True,),
        (_AuthorityIntSubclass(1),),
        (_AuthorityIntCoercible(),),
        (0,),
        (-1,),
        (262_145,),
        tuple(1 for _ in range(129)),
    )
    for bounds in invalid_bounds:
        with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=bounds,
                operation=operation,
                consumer=consumer,
            )

    with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
        arena.consume_admitted_integers(
            result_bit_bounds=(1,),
            operation=object(),
            consumer=consumer,
        )
    with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
        arena.consume_admitted_integers(
            result_bit_bounds=(1,),
            operation=operation,
            consumer=object(),
        )

    assert operation_calls == 0
    assert consumer_calls == 0
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 0


def test_exact_integer_cps_rejects_malformed_or_overbound_results_before_consumer() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    consumer_calls = 0

    class _ResultTupleSubclass(tuple):
        pass

    def consumer(_values: tuple[int, ...]) -> object:
        nonlocal consumer_calls
        consumer_calls += 1
        return object()

    cases = (
        ((2,), lambda: [1]),
        ((2,), lambda: _ResultTupleSubclass((1,))),
        ((2,), lambda: ()),
        ((2,), lambda: (1, 1)),
        ((2,), lambda: (True,)),
        ((2,), lambda: (_AuthorityIntSubclass(1),)),
        ((3,), lambda: (8,)),
        ((3,), lambda: (-8,)),
        ((1, 2), lambda: (1, 4)),
    )
    for bounds, operation in cases:
        with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=bounds,
                operation=operation,
                consumer=consumer,
            )
        assert arena.live_integer_slots == 0

    assert consumer_calls == 0
    assert arena.peak_live_integer_slots == 2


def test_exact_integer_cps_nested_slot_128_succeeds_and_slot_129_is_preoperation_rejected() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    admitted_operations = 0
    forbidden_operations = 0

    def forbidden_operation() -> tuple[int, ...]:
        nonlocal forbidden_operations
        forbidden_operations += 1
        return (0,)

    def descend(remaining: int) -> object:
        nonlocal admitted_operations
        if remaining == 0:
            assert arena.live_integer_slots == 128
            with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
                arena.consume_admitted_integers(
                    result_bit_bounds=(1,),
                    operation=forbidden_operation,
                    consumer=lambda _values: object(),
                )
            assert arena.live_integer_slots == 128
            return "slot-128-held"

        def operation() -> tuple[int, ...]:
            nonlocal admitted_operations
            admitted_operations += 1
            assert arena.live_integer_slots == 129 - remaining
            return (0,)

        return arena.consume_admitted_integers(
            result_bit_bounds=(1,),
            operation=operation,
            consumer=lambda _values: descend(remaining - 1),
        )

    assert descend(128) == "slot-128-held"
    assert admitted_operations == 128
    assert forbidden_operations == 0
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 128

    outer_126 = tuple(0 for _ in range(126))
    inner_2 = (0, 0)
    multi_slot_observations: list[tuple[str, int]] = []

    def consume_inner_at_128(_outer_values: tuple[int, ...]) -> str:
        multi_slot_observations.append(("outer-126-consumer", arena.live_integer_slots))

        def inner_operation() -> tuple[int, ...]:
            multi_slot_observations.append(("inner-2-operation", arena.live_integer_slots))
            return inner_2

        def inner_consumer(_inner_values: tuple[int, ...]) -> str:
            multi_slot_observations.append(("inner-2-consumer", arena.live_integer_slots))
            return "inner-2-admitted"

        result = arena.consume_admitted_integers(
            result_bit_bounds=tuple(1 for _ in range(2)),
            operation=inner_operation,
            consumer=inner_consumer,
        )
        multi_slot_observations.append(("outer-126-resumed", arena.live_integer_slots))
        return result

    assert arena.consume_admitted_integers(
        result_bit_bounds=tuple(1 for _ in range(126)),
        operation=lambda: outer_126,
        consumer=consume_inner_at_128,
    ) == "inner-2-admitted"
    assert multi_slot_observations == [
        ("outer-126-consumer", 126),
        ("inner-2-operation", 128),
        ("inner-2-consumer", 128),
        ("outer-126-resumed", 126),
    ]
    assert arena.live_integer_slots == 0

    outer_127 = tuple(0 for _ in range(127))
    rejected_inner_operations = 0

    def reject_inner_at_129(_outer_values: tuple[int, ...]) -> str:
        nonlocal rejected_inner_operations
        assert arena.live_integer_slots == 127

        def inner_operation() -> tuple[int, ...]:
            nonlocal rejected_inner_operations
            rejected_inner_operations += 1
            return inner_2

        with pytest.raises(ValueError, match="^hopper_numeric_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1, 1),
                operation=inner_operation,
                consumer=lambda _values: object(),
            )
        assert arena.live_integer_slots == 127
        return "inner-2-rejected"

    assert arena.consume_admitted_integers(
        result_bit_bounds=tuple(1 for _ in range(127)),
        operation=lambda: outer_127,
        consumer=reject_inner_at_129,
    ) == "inner-2-rejected"
    assert rejected_inner_operations == 0
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 128


def test_exact_integer_cps_ordinary_exceptions_postseal_propagate_and_unwind() -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)

    class _OperationFailure(RuntimeError):
        pass

    class _ConsumerFailure(RuntimeError):
        pass

    operation_failure_consumer_calls = 0

    def failed_operation() -> tuple[int, ...]:
        assert arena.live_integer_slots == 2
        raise _OperationFailure("operation failed")

    def forbidden_operation_failure_consumer(_values: tuple[int, ...]) -> object:
        nonlocal operation_failure_consumer_calls
        operation_failure_consumer_calls += 1
        return object()

    with pytest.raises(_OperationFailure, match="operation failed"):
        arena.consume_admitted_integers(
            result_bit_bounds=(1, 1),
            operation=failed_operation,
            consumer=forbidden_operation_failure_consumer,
        )
    assert operation_failure_consumer_calls == 0
    assert arena.live_integer_slots == 0

    def failed_consumer(_values: tuple[int, ...]) -> object:
        assert arena.live_integer_slots == 2
        raise _ConsumerFailure("consumer failed")

    with pytest.raises(_ConsumerFailure, match="consumer failed"):
        arena.consume_admitted_integers(
            result_bit_bounds=(1, 1),
            operation=lambda: (0, 1),
            consumer=failed_consumer,
        )
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 2


@pytest.mark.parametrize(
    "critical_type",
    (KeyboardInterrupt, SystemExit, MemoryError),
    ids=("keyboard-interrupt", "system-exit", "memory-error"),
)
def test_exact_integer_cps_critical_exceptions_propagate_and_unwind(
    critical_type,
) -> None:
    module = _authority_module()
    arena = module._HopperExactIntegerArenaV2(module.HOPPER_RESOURCE_AUTHORITY_V2)
    operation_failure_consumer_calls = 0

    def failed_operation() -> tuple[int, ...]:
        assert arena.live_integer_slots == 3
        raise critical_type()

    def forbidden_operation_failure_consumer(_values: tuple[int, ...]) -> object:
        nonlocal operation_failure_consumer_calls
        operation_failure_consumer_calls += 1
        return object()

    with pytest.raises(critical_type):
        arena.consume_admitted_integers(
            result_bit_bounds=(1, 1, 1),
            operation=failed_operation,
            consumer=forbidden_operation_failure_consumer,
        )
    assert operation_failure_consumer_calls == 0
    assert arena.live_integer_slots == 0

    def failed_consumer(_values: tuple[int, ...]) -> object:
        assert arena.live_integer_slots == 3
        raise critical_type()

    with pytest.raises(critical_type):
        arena.consume_admitted_integers(
            result_bit_bounds=(1, 1, 1),
            operation=lambda: (0, 0, 0),
            consumer=failed_consumer,
        )
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 3


def test_exact_integer_cps_authority_drift_overrides_operation_and_consumer_semantics() -> None:
    module = _authority_module()
    authority = module.HOPPER_RESOURCE_AUTHORITY_V2
    arena = module._HopperExactIntegerArenaV2(authority)
    canonical_bits = authority.max_exact_integer_bits
    consumer_calls = 0

    def drifting_operation() -> tuple[int, ...]:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits + 1)
        return (1,)

    def forbidden_consumer(_values: tuple[int, ...]) -> object:
        nonlocal consumer_calls
        consumer_calls += 1
        return object()

    try:
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1,),
                operation=drifting_operation,
                consumer=forbidden_consumer,
            )
    finally:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits)
    assert consumer_calls == 0
    assert arena.live_integer_slots == 0

    class _LosingOperationFailure(RuntimeError):
        pass

    def drifting_failed_operation() -> tuple[int, ...]:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits + 1)
        raise _LosingOperationFailure("authority drift must win")

    try:
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1,),
                operation=drifting_failed_operation,
                consumer=forbidden_consumer,
            )
    finally:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits)
    assert consumer_calls == 0
    assert arena.live_integer_slots == 0

    def drifting_malformed_operation():
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits + 1)
        return [1]

    try:
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1,),
                operation=drifting_malformed_operation,
                consumer=forbidden_consumer,
            )
    finally:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits)
    assert consumer_calls == 0
    assert arena.live_integer_slots == 0

    class _LosingConsumerFailure(RuntimeError):
        pass

    def drifting_consumer(_values: tuple[int, ...]) -> object:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits + 1)
        raise _LosingConsumerFailure("authority drift must win")

    try:
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1,),
                operation=lambda: (1,),
                consumer=drifting_consumer,
            )
    finally:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits)
    assert arena.live_integer_slots == 0

    def drifting_returning_consumer(_values: tuple[int, ...]) -> object:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits + 1)
        return object()

    try:
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(1,),
                operation=lambda: (1,),
                consumer=drifting_returning_consumer,
            )
    finally:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits)
    assert arena.live_integer_slots == 0

    entry_operation_calls = 0

    def forbidden_entry_operation() -> tuple[int, ...]:
        nonlocal entry_operation_calls
        entry_operation_calls += 1
        return ()

    object.__setattr__(authority, "max_exact_integer_bits", canonical_bits + 1)
    try:
        with pytest.raises(ValueError, match="^hopper_authority_contract_mismatch$"):
            arena.consume_admitted_integers(
                result_bit_bounds=(),
                operation=forbidden_entry_operation,
                consumer=forbidden_consumer,
            )
    finally:
        object.__setattr__(authority, "max_exact_integer_bits", canonical_bits)
    assert entry_operation_calls == 0
    assert consumer_calls == 0
    assert arena.live_integer_slots == 0
    assert arena.peak_live_integer_slots == 1
