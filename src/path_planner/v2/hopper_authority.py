from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from inspect import getsource
from math import isfinite
from types import FunctionType
from typing import TypeVar

import path_planner.v2.ballistics as _ballistics
from path_planner.v2.contracts import PlatformKindV2
from path_planner.v2.profiles import (
    HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2,
    HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
    HopperProfileV2,
    PlatformProfileV2,
    audit_hopper_profile_v2,
)


HOPPER_MAX_BALLISTIC_SAMPLES_V2 = 100_000
HOPPER_MAX_LANDING_CANDIDATES_V2 = 1_000_000
HOPPER_MAX_REPLAY_STEPS_V2 = 100_000
HOPPER_EXACT_MAX_INTEGER_BITS_V2 = 262_144
HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2 = 128
HOPPER_HARD_ACCOUNTED_MEMORY_BYTES_V2 = 536_870_912
HOPPER_SEARCH_BASE_BYTES_V2 = 4_096
HOPPER_SEARCH_RECORD_BYTES_V2 = 1_024
HOPPER_BALLISTIC_BUILD_BASE_BYTES_V2 = 4_096
HOPPER_BALLISTIC_BUILD_PER_SAMPLE_BYTES_V2 = 96
HOPPER_RETURNED_BALLISTIC_SAMPLE_BYTES_V2 = 48
HOPPER_LANDING_BUILD_BASE_BYTES_V2 = 65_536
HOPPER_LANDING_BUILD_PER_CANDIDATE_BYTES_V2 = 384
HOPPER_EXACT_BASE_BYTES_V2 = 4_096
HOPPER_EXACT_INTEGER_SLOT_OVERHEAD_BYTES_V2 = 16
HOPPER_EXACT_DISTINCT_CELL_BYTES_V2 = 320
HOPPER_ROUTE_BUILD_BASE_BYTES_V2 = 4_096
HOPPER_ROUTE_PRIMITIVE_BYTES_V2 = 512

HOPPER_BALLISTIC_BUILD_MAX_BYTES_V2 = HOPPER_BALLISTIC_BUILD_BASE_BYTES_V2 + (
    HOPPER_BALLISTIC_BUILD_PER_SAMPLE_BYTES_V2 * HOPPER_MAX_BALLISTIC_SAMPLES_V2
)
HOPPER_LANDING_BUILD_MAX_BYTES_V2 = HOPPER_LANDING_BUILD_BASE_BYTES_V2 + (
    HOPPER_LANDING_BUILD_PER_CANDIDATE_BYTES_V2
    * HOPPER_MAX_LANDING_CANDIDATES_V2
)
HOPPER_EXACT_INTEGER_SLOT_BYTES_V2 = (
    HOPPER_EXACT_INTEGER_SLOT_OVERHEAD_BYTES_V2
    + (HOPPER_EXACT_MAX_INTEGER_BITS_V2 + 7) // 8
)
HOPPER_EXACT_ORACLE_MAX_BYTES_V2 = (
    HOPPER_EXACT_BASE_BYTES_V2
    + HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2 * HOPPER_EXACT_INTEGER_SLOT_BYTES_V2
    + HOPPER_MAX_REPLAY_STEPS_V2 * HOPPER_EXACT_DISTINCT_CELL_BYTES_V2
)

_BALLISTIC_HELPER_ID = "sample_hopper_ballistic_arc_capped/v1"
_LANDING_HELPER_ID = "landing_zone_cells_capped/v1"
_MEMORY_ACCOUNTING_ID = "hopper_deterministic_admission_bytes/v1"
_RESOURCE_SCHEMA_VERSION = "hopper-resource-authority/v1"
_RESOURCE_LINEAGE_SCHEMA_VERSION = "hopper-resource-authority-lineage/v1"

_TRUSTED_BALLISTIC_HELPER = _ballistics._sample_hopper_ballistic_arc_capped_v2
_TRUSTED_LANDING_HELPER = _ballistics.landing_zone_cells_capped_v2

_NUMERIC_FIELD_VALUES = (
    ("max_ballistic_samples", HOPPER_MAX_BALLISTIC_SAMPLES_V2),
    ("max_landing_candidates", HOPPER_MAX_LANDING_CANDIDATES_V2),
    ("max_replay_steps", HOPPER_MAX_REPLAY_STEPS_V2),
    ("max_exact_integer_bits", HOPPER_EXACT_MAX_INTEGER_BITS_V2),
    ("max_exact_live_integer_slots", HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2),
    ("hard_accounted_memory_bytes", HOPPER_HARD_ACCOUNTED_MEMORY_BYTES_V2),
    ("search_base_bytes", HOPPER_SEARCH_BASE_BYTES_V2),
    ("search_record_bytes", HOPPER_SEARCH_RECORD_BYTES_V2),
    ("ballistic_build_base_bytes", HOPPER_BALLISTIC_BUILD_BASE_BYTES_V2),
    (
        "ballistic_build_per_sample_bytes",
        HOPPER_BALLISTIC_BUILD_PER_SAMPLE_BYTES_V2,
    ),
    (
        "ballistic_retained_per_sample_bytes",
        HOPPER_RETURNED_BALLISTIC_SAMPLE_BYTES_V2,
    ),
    ("landing_build_base_bytes", HOPPER_LANDING_BUILD_BASE_BYTES_V2),
    (
        "landing_build_per_candidate_bytes",
        HOPPER_LANDING_BUILD_PER_CANDIDATE_BYTES_V2,
    ),
    ("exact_base_bytes", HOPPER_EXACT_BASE_BYTES_V2),
    (
        "exact_integer_slot_overhead_bytes",
        HOPPER_EXACT_INTEGER_SLOT_OVERHEAD_BYTES_V2,
    ),
    ("exact_distinct_cell_bytes", HOPPER_EXACT_DISTINCT_CELL_BYTES_V2),
    ("route_build_base_bytes", HOPPER_ROUTE_BUILD_BASE_BYTES_V2),
    ("route_primitive_bytes", HOPPER_ROUTE_PRIMITIVE_BYTES_V2),
)
_STRING_FIELD_VALUES = (
    ("ballistic_helper_id", _BALLISTIC_HELPER_ID),
    ("landing_helper_id", _LANDING_HELPER_ID),
    ("memory_accounting_id", _MEMORY_ACCOUNTING_ID),
    ("schema_version", _RESOURCE_SCHEMA_VERSION),
)
_RESOURCE_FIELD_NAMES = (
    "ballistic_helper",
    "ballistic_helper_id",
    "landing_helper",
    "landing_helper_id",
    "memory_accounting_id",
    *(name for name, _value in _NUMERIC_FIELD_VALUES),
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class HopperResourceAuthorityV2:
    ballistic_helper: Callable[..., tuple[_ballistics.BallisticSampleV2, ...]]
    ballistic_helper_id: str
    landing_helper: Callable[..., tuple[_ballistics.LandingCellMassV2, ...]]
    landing_helper_id: str
    memory_accounting_id: str
    max_ballistic_samples: int
    max_landing_candidates: int
    max_replay_steps: int
    max_exact_integer_bits: int
    max_exact_live_integer_slots: int
    hard_accounted_memory_bytes: int
    search_base_bytes: int
    search_record_bytes: int
    ballistic_build_base_bytes: int
    ballistic_build_per_sample_bytes: int
    ballistic_retained_per_sample_bytes: int
    landing_build_base_bytes: int
    landing_build_per_candidate_bytes: int
    exact_base_bytes: int
    exact_integer_slot_overhead_bytes: int
    exact_distinct_cell_bytes: int
    route_build_base_bytes: int
    route_primitive_bytes: int
    schema_version: str

    def __post_init__(self) -> None:
        if self.ballistic_helper is not _TRUSTED_BALLISTIC_HELPER:
            raise ValueError("ballistic_helper must be the trusted capped helper")
        if self.landing_helper is not _TRUSTED_LANDING_HELPER:
            raise ValueError("landing_helper must be the trusted capped helper")
        for field_name, expected in _STRING_FIELD_VALUES:
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"{field_name} must be exact str")
            if value != expected:
                raise ValueError(f"{field_name} must be exactly {expected}")
        for field_name, expected in _NUMERIC_FIELD_VALUES:
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be exact int")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            if value != expected:
                raise ValueError(f"{field_name} must be exactly {expected}")


def _resource_authority_record_is_exact_v2(authority: object) -> bool:
    if type(authority) is not HopperResourceAuthorityV2:
        return False
    try:
        if authority.ballistic_helper is not _TRUSTED_BALLISTIC_HELPER:
            return False
        if authority.landing_helper is not _TRUSTED_LANDING_HELPER:
            return False
        for field_name, expected in _STRING_FIELD_VALUES:
            value = getattr(authority, field_name)
            if type(value) is not str or value != expected:
                return False
        for field_name, expected in _NUMERIC_FIELD_VALUES:
            value = getattr(authority, field_name)
            if type(value) is not int or value != expected:
                return False
    except AttributeError:
        return False
    return True


HOPPER_RESOURCE_AUTHORITY_V2 = HopperResourceAuthorityV2(
    ballistic_helper=_TRUSTED_BALLISTIC_HELPER,
    ballistic_helper_id=_BALLISTIC_HELPER_ID,
    landing_helper=_TRUSTED_LANDING_HELPER,
    landing_helper_id=_LANDING_HELPER_ID,
    memory_accounting_id=_MEMORY_ACCOUNTING_ID,
    max_ballistic_samples=HOPPER_MAX_BALLISTIC_SAMPLES_V2,
    max_landing_candidates=HOPPER_MAX_LANDING_CANDIDATES_V2,
    max_replay_steps=HOPPER_MAX_REPLAY_STEPS_V2,
    max_exact_integer_bits=HOPPER_EXACT_MAX_INTEGER_BITS_V2,
    max_exact_live_integer_slots=HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2,
    hard_accounted_memory_bytes=HOPPER_HARD_ACCOUNTED_MEMORY_BYTES_V2,
    search_base_bytes=HOPPER_SEARCH_BASE_BYTES_V2,
    search_record_bytes=HOPPER_SEARCH_RECORD_BYTES_V2,
    ballistic_build_base_bytes=HOPPER_BALLISTIC_BUILD_BASE_BYTES_V2,
    ballistic_build_per_sample_bytes=HOPPER_BALLISTIC_BUILD_PER_SAMPLE_BYTES_V2,
    ballistic_retained_per_sample_bytes=HOPPER_RETURNED_BALLISTIC_SAMPLE_BYTES_V2,
    landing_build_base_bytes=HOPPER_LANDING_BUILD_BASE_BYTES_V2,
    landing_build_per_candidate_bytes=HOPPER_LANDING_BUILD_PER_CANDIDATE_BYTES_V2,
    exact_base_bytes=HOPPER_EXACT_BASE_BYTES_V2,
    exact_integer_slot_overhead_bytes=HOPPER_EXACT_INTEGER_SLOT_OVERHEAD_BYTES_V2,
    exact_distinct_cell_bytes=HOPPER_EXACT_DISTINCT_CELL_BYTES_V2,
    route_build_base_bytes=HOPPER_ROUTE_BUILD_BASE_BYTES_V2,
    route_primitive_bytes=HOPPER_ROUTE_PRIMITIVE_BYTES_V2,
    schema_version=_RESOURCE_SCHEMA_VERSION,
)


def _hopper_resource_authority_in_memory_token_v2(
    authority: HopperResourceAuthorityV2,
) -> tuple[object, ...]:
    if not _resource_authority_record_is_exact_v2(authority):
        raise ValueError("hopper_authority_contract_mismatch")
    return tuple(getattr(authority, name) for name in _RESOURCE_FIELD_NAMES)


def _hopper_resource_authority_lineage_token_v2(
    authority: HopperResourceAuthorityV2,
) -> tuple[object, ...]:
    if not _resource_authority_record_is_exact_v2(authority):
        raise ValueError("hopper_authority_contract_mismatch")
    return (
        _RESOURCE_LINEAGE_SCHEMA_VERSION,
        authority.ballistic_helper_id,
        authority.landing_helper_id,
        authority.memory_accounting_id,
        *(getattr(authority, name) for name, _value in _NUMERIC_FIELD_VALUES),
        authority.schema_version,
    )


HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2 = (
    _hopper_resource_authority_in_memory_token_v2(HOPPER_RESOURCE_AUTHORITY_V2)
)
HOPPER_RESOURCE_AUTHORITY_LINEAGE_TOKEN_V2 = (
    _hopper_resource_authority_lineage_token_v2(HOPPER_RESOURCE_AUTHORITY_V2)
)

_CANONICAL_RESOURCE_AUTHORITY = HOPPER_RESOURCE_AUTHORITY_V2
_CANONICAL_IN_MEMORY_TOKEN = HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2
_CANONICAL_LINEAGE_TOKEN = HOPPER_RESOURCE_AUTHORITY_LINEAGE_TOKEN_V2
_MODULE_INTEGER_CONSTANT_VALUES = (
    ("HOPPER_MAX_BALLISTIC_SAMPLES_V2", 100_000),
    ("HOPPER_MAX_LANDING_CANDIDATES_V2", 1_000_000),
    ("HOPPER_MAX_REPLAY_STEPS_V2", 100_000),
    ("HOPPER_EXACT_MAX_INTEGER_BITS_V2", 262_144),
    ("HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2", 128),
    ("HOPPER_HARD_ACCOUNTED_MEMORY_BYTES_V2", 536_870_912),
    ("HOPPER_SEARCH_BASE_BYTES_V2", 4_096),
    ("HOPPER_SEARCH_RECORD_BYTES_V2", 1_024),
    ("HOPPER_BALLISTIC_BUILD_BASE_BYTES_V2", 4_096),
    ("HOPPER_BALLISTIC_BUILD_PER_SAMPLE_BYTES_V2", 96),
    ("HOPPER_RETURNED_BALLISTIC_SAMPLE_BYTES_V2", 48),
    ("HOPPER_LANDING_BUILD_BASE_BYTES_V2", 65_536),
    ("HOPPER_LANDING_BUILD_PER_CANDIDATE_BYTES_V2", 384),
    ("HOPPER_EXACT_BASE_BYTES_V2", 4_096),
    ("HOPPER_EXACT_INTEGER_SLOT_OVERHEAD_BYTES_V2", 16),
    ("HOPPER_EXACT_DISTINCT_CELL_BYTES_V2", 320),
    ("HOPPER_ROUTE_BUILD_BASE_BYTES_V2", 4_096),
    ("HOPPER_ROUTE_PRIMITIVE_BYTES_V2", 512),
    ("HOPPER_BALLISTIC_BUILD_MAX_BYTES_V2", 9_604_096),
    ("HOPPER_LANDING_BUILD_MAX_BYTES_V2", 384_065_536),
    ("HOPPER_EXACT_INTEGER_SLOT_BYTES_V2", 32_784),
    ("HOPPER_EXACT_ORACLE_MAX_BYTES_V2", 36_200_448),
)


def _require_canonical_resource_authority_v2(
    authority: HopperResourceAuthorityV2,
) -> None:
    if authority is not _CANONICAL_RESOURCE_AUTHORITY:
        raise ValueError("hopper_authority_contract_mismatch")
    if HOPPER_RESOURCE_AUTHORITY_V2 is not _CANONICAL_RESOURCE_AUTHORITY:
        raise ValueError("hopper_authority_contract_mismatch")
    if not _resource_authority_record_is_exact_v2(authority):
        raise ValueError("hopper_authority_contract_mismatch")
    if (
        getattr(_ballistics, "_sample_hopper_ballistic_arc_capped_v2", None)
        is not _TRUSTED_BALLISTIC_HELPER
        or getattr(_ballistics, "landing_zone_cells_capped_v2", None)
        is not _TRUSTED_LANDING_HELPER
    ):
        raise ValueError("hopper_authority_contract_mismatch")
    for constant_name, expected in _MODULE_INTEGER_CONSTANT_VALUES:
        value = globals().get(constant_name)
        if type(value) is not int or value != expected:
            raise ValueError("hopper_authority_contract_mismatch")
    if (
        HOPPER_RESOURCE_AUTHORITY_IN_MEMORY_TOKEN_V2
        is not _CANONICAL_IN_MEMORY_TOKEN
        or HOPPER_RESOURCE_AUTHORITY_LINEAGE_TOKEN_V2 is not _CANONICAL_LINEAGE_TOKEN
    ):
        raise ValueError("hopper_authority_contract_mismatch")


def _call_captured_ballistic_helper_v2(
    authority: HopperResourceAuthorityV2,
    start: _ballistics.BallisticStartV2,
    speed_mps: float,
    elevation_rad: float,
    azimuth_index: int,
    g_mps2: float,
) -> tuple[_ballistics.BallisticSampleV2, ...]:
    _require_canonical_resource_authority_v2(authority)
    helper = authority.ballistic_helper
    cap = authority.max_ballistic_samples
    try:
        result = helper(
            start,
            speed_mps,
            elevation_rad,
            azimuth_index,
            g_mps2,
            max_sample_count=cap,
        )
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        _require_canonical_resource_authority_v2(authority)
        raise
    _require_canonical_resource_authority_v2(authority)
    return result


def _call_captured_landing_helper_v2(
    authority: HopperResourceAuthorityV2,
    mean_xy: object,
    sigma_m: float,
    probability_threshold: float,
    geometry: object,
) -> tuple[_ballistics.LandingCellMassV2, ...]:
    _require_canonical_resource_authority_v2(authority)
    helper = authority.landing_helper
    cap = authority.max_landing_candidates
    try:
        result = helper(
            mean_xy,
            sigma_m,
            probability_threshold,
            geometry,
            max_candidate_count=cap,
        )
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        _require_canonical_resource_authority_v2(authority)
        raise
    _require_canonical_resource_authority_v2(authority)
    return result


_T = TypeVar("_T")


class _HopperExactIntegerArenaV2:
    __slots__ = ("_authority", "_live_integer_slots", "_peak_live_integer_slots")

    def __init__(self, authority: HopperResourceAuthorityV2) -> None:
        _require_canonical_resource_authority_v2(authority)
        self._authority = authority
        self._live_integer_slots = 0
        self._peak_live_integer_slots = 0

    @property
    def live_integer_slots(self) -> int:
        return self._live_integer_slots

    @property
    def peak_live_integer_slots(self) -> int:
        return self._peak_live_integer_slots

    def run_admitted(
        self,
        *,
        result_bit_bound: int,
        operation: Callable[[], _T],
    ) -> _T:
        _require_canonical_resource_authority_v2(self._authority)
        if (
            type(result_bit_bound) is not int
            or result_bit_bound <= 0
            or result_bit_bound > self._authority.max_exact_integer_bits
        ):
            raise ValueError("hopper_numeric_contract_mismatch")
        prospective_live_slots = self._live_integer_slots + 1
        if prospective_live_slots > self._authority.max_exact_live_integer_slots:
            raise ValueError("hopper_numeric_contract_mismatch")
        if not callable(operation):
            raise ValueError("hopper_numeric_contract_mismatch")

        self._live_integer_slots = prospective_live_slots
        if prospective_live_slots > self._peak_live_integer_slots:
            self._peak_live_integer_slots = prospective_live_slots
        try:
            result = operation()
        except (KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:
            _require_canonical_resource_authority_v2(self._authority)
            raise
        finally:
            self._live_integer_slots -= 1
        _require_canonical_resource_authority_v2(self._authority)
        return result

    def consume_admitted_integers(
        self,
        *,
        result_bit_bounds: tuple[int, ...],
        operation: Callable[[], tuple[int, ...]],
        consumer: Callable[[tuple[int, ...]], _T],
    ) -> _T:
        _require_canonical_resource_authority_v2(self._authority)
        if type(result_bit_bounds) is not tuple or not result_bit_bounds:
            raise ValueError("hopper_numeric_contract_mismatch")
        if any(
            type(bit_bound) is not int
            or bit_bound <= 0
            or bit_bound > self._authority.max_exact_integer_bits
            for bit_bound in result_bit_bounds
        ):
            raise ValueError("hopper_numeric_contract_mismatch")
        if not callable(operation) or not callable(consumer):
            raise ValueError("hopper_numeric_contract_mismatch")

        required_slots = len(result_bit_bounds)
        available_slots = (
            self._authority.max_exact_live_integer_slots
            - self._live_integer_slots
        )
        if required_slots > available_slots:
            raise ValueError("hopper_numeric_contract_mismatch")

        self._live_integer_slots += required_slots
        if self._live_integer_slots > self._peak_live_integer_slots:
            self._peak_live_integer_slots = self._live_integer_slots
        try:
            try:
                values = operation()
            except (KeyboardInterrupt, MemoryError, SystemExit):
                raise
            except Exception:
                _require_canonical_resource_authority_v2(self._authority)
                raise

            _require_canonical_resource_authority_v2(self._authority)
            if type(values) is not tuple or len(values) != required_slots:
                raise ValueError("hopper_numeric_contract_mismatch")
            if any(
                type(value) is not int or value.bit_length() > bit_bound
                for value, bit_bound in zip(values, result_bit_bounds, strict=True)
            ):
                raise ValueError("hopper_numeric_contract_mismatch")

            try:
                result = consumer(values)
            except (KeyboardInterrupt, MemoryError, SystemExit):
                raise
            except Exception:
                _require_canonical_resource_authority_v2(self._authority)
                raise
            _require_canonical_resource_authority_v2(self._authority)
            return result
        finally:
            self._live_integer_slots -= required_slots


_HOPPER_PROVIDER_AUTHORITY_SCHEMA_VERSION_V2 = "hopper-provider-authority/v1"
_HOPPER_PARAMETER_SET_SCHEMA_VERSION_V2 = "hopper-parameter-set-record/v1"
_HOPPER_PARAMETER_SET_ABSENT_SCHEMA_VERSION_V2 = (
    "hopper-parameter-set-absent/v1"
)
_HOPPER_GATE5B_PARAMETER_SET_ID_V1 = "hopper_gate5b_algorithm_fixture/v1"
_HOPPER_GATE5B_BASE_PROFILE_ID_V1 = (
    "hopper-lunar-ballistic-gate5b-fixture/v1"
)
_HOPPER_GATE5B_STOP_CONDITION_ID_V1 = (
    "hopper_same_height_nominal_recenter_capture_le_3mps_"
    "stop_simulation_proxy/v1"
)
_HOPPER_GATE5B_ENERGY_MODEL_ID_V1 = (
    "hopper_launch_speed_squared_relative_energy/v1"
)
_HOPPER_GATE5B_BODY_ENVELOPE_RADIUS_M = 0.25
_HOPPER_GATE5B_LAUNCH_REFERENCE_HEIGHT_M = 0.50
_HOPPER_GATE5B_ARC_CLEARANCE_MARGIN_M = 0.10
_HOPPER_GATE5B_LANDING_FOOTPRINT_RADIUS_M = 0.30
HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_PARAMETER_SET_ID_V1 = (
    "hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1"
)
_HOPPER_GENERIC_INTERNAL_BASE_PROFILE_ID_V1 = (
    "hopper-generic-internal-computational-simulation-proxy-midterm/v1"
)
_HOPPER_GENERIC_INTERNAL_STOP_CONDITION_ID_V1 = (
    "hopper_generic_internal_same_height_capture_le_2p5mps_"
    "stop_computational_simulation_proxy/v1"
)
_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_ID_V1 = (
    "hopper_generic_internal_stop_evaluator_exact_binary64/v1"
)
_HOPPER_GENERIC_INTERNAL_ENERGY_MODEL_ID_V1 = (
    "hopper_generic_internal_launch_speed_squared_normalized_2p5mps_"
    "relative_energy_computational_simulation_proxy/v1"
)
_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_ID_V1 = (
    "hopper_generic_internal_relative_energy_evaluator_exact_binary64/v1"
)
_HOPPER_GENERIC_INTERNAL_EVIDENCE_CLASS_V1 = (
    "project_internal_generic_computational_simulation_proxy/v1"
)
_HOPPER_GENERIC_INTERNAL_IMPLEMENTATION_SCHEMA_VERSION_V1 = (
    "hopper-generic-internal-computational-simulation-proxy-"
    "implementation-record/v1"
)
_HOPPER_GENERIC_INTERNAL_BODY_ENVELOPE_RADIUS_M = 0.375
_HOPPER_GENERIC_INTERNAL_LAUNCH_REFERENCE_HEIGHT_M = 0.750
_HOPPER_GENERIC_INTERNAL_ARC_CLEARANCE_MARGIN_M = 0.125
_HOPPER_GENERIC_INTERNAL_LANDING_FOOTPRINT_RADIUS_M = 0.625


def _fixture_speed_v2(speed_mps: float) -> float:
    if type(speed_mps) is not float:
        raise TypeError("speed_mps must be exact float")
    if not isfinite(speed_mps) or speed_mps <= 0.0:
        raise ValueError("speed_mps must be finite and positive")
    return speed_mps


def _hopper_gate5b_stop_evaluator_v1(speed_mps: float) -> bool:
    return _fixture_speed_v2(speed_mps) <= 3.0


def _hopper_gate5b_energy_evaluator_v1(speed_mps: float) -> float:
    ratio = _fixture_speed_v2(speed_mps) / 3.0
    return ratio * ratio


_TRUSTED_HOPPER_GATE5B_STOP_EVALUATOR_V1 = _hopper_gate5b_stop_evaluator_v1
_TRUSTED_HOPPER_GATE5B_ENERGY_EVALUATOR_V1 = _hopper_gate5b_energy_evaluator_v1


def _hopper_generic_internal_stop_evaluator_v1(speed_mps: float) -> bool:
    if type(speed_mps) is not float:
        raise TypeError("speed_mps must be exact float")
    if not isfinite(speed_mps) or speed_mps <= 0.0:
        raise ValueError("speed_mps must be finite and positive")
    return speed_mps <= 2.5


def _hopper_generic_internal_energy_evaluator_v1(speed_mps: float) -> float:
    if type(speed_mps) is not float:
        raise TypeError("speed_mps must be exact float")
    if not isfinite(speed_mps) or speed_mps <= 0.0:
        raise ValueError("speed_mps must be finite and positive")
    ratio = speed_mps / 2.5
    return ratio * ratio


def _evaluator_source_sha256_v2(evaluator: object) -> str:
    if type(evaluator) is not FunctionType:
        raise TypeError("evaluator must be an exact Python function")
    return sha256(getsource(evaluator).encode("utf-8")).hexdigest()


_TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_V1 = (
    _hopper_generic_internal_stop_evaluator_v1
)
_TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_V1 = (
    _hopper_generic_internal_energy_evaluator_v1
)
_TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_CODE_V1 = (
    _TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_V1.__code__
)
_TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_CODE_V1 = (
    _TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_V1.__code__
)
_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_SOURCE_SHA256_V1 = (
    _evaluator_source_sha256_v2(
        _TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_V1
    )
)
_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_SOURCE_SHA256_V1 = (
    _evaluator_source_sha256_v2(
        _TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_V1
    )
)


@dataclass(frozen=True, slots=True)
class HopperProviderAuthorityV2:
    hopper_profile: HopperProfileV2
    parameter_set_id: str | None
    authority_schema_version: str

    def __post_init__(self) -> None:
        if type(self.hopper_profile) is not HopperProfileV2:
            raise TypeError("hopper_profile must be exact HopperProfileV2")
        if self.parameter_set_id is not None:
            if type(self.parameter_set_id) is not str:
                raise TypeError("parameter_set_id must be exact str or None")
            if not self.parameter_set_id:
                raise ValueError("parameter_set_id must be nonempty when provided")
        if type(self.authority_schema_version) is not str:
            raise TypeError("authority_schema_version must be exact str")
        if (
            self.authority_schema_version
            != _HOPPER_PROVIDER_AUTHORITY_SCHEMA_VERSION_V2
        ):
            raise ValueError(
                "authority_schema_version must be "
                f"{_HOPPER_PROVIDER_AUTHORITY_SCHEMA_VERSION_V2}"
            )


@dataclass(frozen=True, slots=True)
class HopperParameterSetRecordV2:
    parameter_set_id: str
    base_profile_id: str
    body_envelope_radius_m: float
    launch_reference_height_m: float
    arc_clearance_margin_m: float
    landing_footprint_radius_m: float
    stop_condition: str
    energy_model: str
    stop_evaluator: Callable[[float], bool]
    energy_evaluator: Callable[[float], float]
    evidence_class: str
    simulation_proxy: bool
    formal_evidence_eligible: bool
    schema_version: str

    def __post_init__(self) -> None:
        expected_strings = (
            ("parameter_set_id", _HOPPER_GATE5B_PARAMETER_SET_ID_V1),
            ("base_profile_id", _HOPPER_GATE5B_BASE_PROFILE_ID_V1),
            ("stop_condition", _HOPPER_GATE5B_STOP_CONDITION_ID_V1),
            ("energy_model", _HOPPER_GATE5B_ENERGY_MODEL_ID_V1),
            ("evidence_class", "test_fixture"),
            ("schema_version", _HOPPER_PARAMETER_SET_SCHEMA_VERSION_V2),
        )
        for name, expected in expected_strings:
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name} must be exact str")
            if not value:
                raise ValueError(f"{name} must be nonempty")
            if value != expected:
                raise ValueError(f"{name} must be exactly {expected}")

        expected_floats = (
            ("body_envelope_radius_m", _HOPPER_GATE5B_BODY_ENVELOPE_RADIUS_M),
            (
                "launch_reference_height_m",
                _HOPPER_GATE5B_LAUNCH_REFERENCE_HEIGHT_M,
            ),
            ("arc_clearance_margin_m", _HOPPER_GATE5B_ARC_CLEARANCE_MARGIN_M),
            (
                "landing_footprint_radius_m",
                _HOPPER_GATE5B_LANDING_FOOTPRINT_RADIUS_M,
            ),
        )
        for name, expected in expected_floats:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be exact float")
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            if value.hex() != expected.hex():
                raise ValueError(f"{name} must be the frozen fixture value")

        if self.stop_evaluator is not _TRUSTED_HOPPER_GATE5B_STOP_EVALUATOR_V1:
            raise ValueError("stop_evaluator must be the trusted fixture evaluator")
        if self.energy_evaluator is not _TRUSTED_HOPPER_GATE5B_ENERGY_EVALUATOR_V1:
            raise ValueError("energy_evaluator must be the trusted fixture evaluator")
        if type(self.simulation_proxy) is not bool:
            raise TypeError("simulation_proxy must be exact bool")
        if self.simulation_proxy is not True:
            raise ValueError("simulation_proxy must be exactly True")
        if type(self.formal_evidence_eligible) is not bool:
            raise TypeError("formal_evidence_eligible must be exact bool")
        if self.formal_evidence_eligible is not False:
            raise ValueError("formal_evidence_eligible must be exactly False")


@dataclass(frozen=True, slots=True)
class HopperGenericInternalSimulationProxyImplementationRecordV2:
    parameter_set_id: str
    base_profile_id: str
    capability_revision: str
    body_envelope_radius_m: float
    launch_reference_height_m: float
    arc_clearance_margin_m: float
    landing_footprint_radius_m: float
    stop_condition: str
    stop_evaluator_id: str
    stop_evaluator_source_sha256: str
    energy_model: str
    energy_evaluator_id: str
    energy_evaluator_source_sha256: str
    stop_evaluator: Callable[[float], bool]
    energy_evaluator: Callable[[float], float]
    evidence_class: str
    simulation_proxy: bool
    physical_capability_claimed: bool
    hardware_certification_claimed: bool
    formal_evidence_eligible: bool
    schema_version: str

    def __post_init__(self) -> None:
        expected_strings = (
            (
                "parameter_set_id",
                HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_PARAMETER_SET_ID_V1,
            ),
            ("base_profile_id", _HOPPER_GENERIC_INTERNAL_BASE_PROFILE_ID_V1),
            (
                "capability_revision",
                HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2,
            ),
            ("stop_condition", _HOPPER_GENERIC_INTERNAL_STOP_CONDITION_ID_V1),
            (
                "stop_evaluator_id",
                _HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_ID_V1,
            ),
            (
                "stop_evaluator_source_sha256",
                _HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_SOURCE_SHA256_V1,
            ),
            ("energy_model", _HOPPER_GENERIC_INTERNAL_ENERGY_MODEL_ID_V1),
            (
                "energy_evaluator_id",
                _HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_ID_V1,
            ),
            (
                "energy_evaluator_source_sha256",
                _HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_SOURCE_SHA256_V1,
            ),
            ("evidence_class", _HOPPER_GENERIC_INTERNAL_EVIDENCE_CLASS_V1),
            (
                "schema_version",
                _HOPPER_GENERIC_INTERNAL_IMPLEMENTATION_SCHEMA_VERSION_V1,
            ),
        )
        for name, expected in expected_strings:
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name} must be exact str")
            if value != expected:
                raise ValueError(f"{name} must be exactly {expected}")

        expected_floats = (
            (
                "body_envelope_radius_m",
                _HOPPER_GENERIC_INTERNAL_BODY_ENVELOPE_RADIUS_M,
            ),
            (
                "launch_reference_height_m",
                _HOPPER_GENERIC_INTERNAL_LAUNCH_REFERENCE_HEIGHT_M,
            ),
            (
                "arc_clearance_margin_m",
                _HOPPER_GENERIC_INTERNAL_ARC_CLEARANCE_MARGIN_M,
            ),
            (
                "landing_footprint_radius_m",
                _HOPPER_GENERIC_INTERNAL_LANDING_FOOTPRINT_RADIUS_M,
            ),
        )
        for name, expected in expected_floats:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be exact float")
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            if value.hex() != expected.hex():
                raise ValueError(f"{name} must be the frozen internal proxy value")

        if (
            self.body_envelope_radius_m + self.arc_clearance_margin_m
            != 0.5
            or self.launch_reference_height_m != 0.5 + 0.25
            or self.landing_footprint_radius_m
            != self.body_envelope_radius_m + 0.25
        ):
            raise ValueError("generic internal proxy derivation contract mismatch")
        if (
            self.stop_evaluator
            is not _TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_V1
        ):
            raise ValueError("stop_evaluator must be the trusted internal evaluator")
        if (
            self.energy_evaluator
            is not _TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_V1
        ):
            raise ValueError(
                "energy_evaluator must be the trusted internal evaluator"
            )
        expected_flags = (
            ("simulation_proxy", True),
            ("physical_capability_claimed", False),
            ("hardware_certification_claimed", False),
            ("formal_evidence_eligible", False),
        )
        for name, expected in expected_flags:
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be exact bool")
            if value is not expected:
                raise ValueError(f"{name} must be exactly {expected}")


def hopper_gate5b_algorithm_fixture_v1() -> HopperProfileV2:
    return HopperProfileV2(
        profile=PlatformProfileV2(
            profile_id=_HOPPER_GATE5B_BASE_PROFILE_ID_V1,
            platform_kind=PlatformKindV2.HOPPER,
            capability_revision=HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
            goal_position_tolerance_m=0.0,
            goal_heading_tolerance_rad=0.0,
        ),
        body_envelope_radius_m=_HOPPER_GATE5B_BODY_ENVELOPE_RADIUS_M,
        launch_reference_height_m=_HOPPER_GATE5B_LAUNCH_REFERENCE_HEIGHT_M,
        arc_clearance_margin_m=_HOPPER_GATE5B_ARC_CLEARANCE_MARGIN_M,
        landing_footprint_radius_m=_HOPPER_GATE5B_LANDING_FOOTPRINT_RADIUS_M,
        stop_condition=_HOPPER_GATE5B_STOP_CONDITION_ID_V1,
        energy_model=_HOPPER_GATE5B_ENERGY_MODEL_ID_V1,
    )


def hopper_generic_internal_simulation_proxy_midterm_v1() -> HopperProfileV2:
    return HopperProfileV2(
        profile=PlatformProfileV2(
            profile_id=_HOPPER_GENERIC_INTERNAL_BASE_PROFILE_ID_V1,
            platform_kind=PlatformKindV2.HOPPER,
            capability_revision=(
                HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2
            ),
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
            goal_position_tolerance_m=0.0,
            goal_heading_tolerance_rad=0.0,
        ),
        body_envelope_radius_m=(
            _HOPPER_GENERIC_INTERNAL_BODY_ENVELOPE_RADIUS_M
        ),
        launch_reference_height_m=(
            _HOPPER_GENERIC_INTERNAL_LAUNCH_REFERENCE_HEIGHT_M
        ),
        arc_clearance_margin_m=(
            _HOPPER_GENERIC_INTERNAL_ARC_CLEARANCE_MARGIN_M
        ),
        landing_footprint_radius_m=(
            _HOPPER_GENERIC_INTERNAL_LANDING_FOOTPRINT_RADIUS_M
        ),
        stop_condition=_HOPPER_GENERIC_INTERNAL_STOP_CONDITION_ID_V1,
        energy_model=_HOPPER_GENERIC_INTERNAL_ENERGY_MODEL_ID_V1,
    )


HOPPER_GATE5B_ALGORITHM_FIXTURE_V1 = HopperParameterSetRecordV2(
    parameter_set_id=_HOPPER_GATE5B_PARAMETER_SET_ID_V1,
    base_profile_id=_HOPPER_GATE5B_BASE_PROFILE_ID_V1,
    body_envelope_radius_m=_HOPPER_GATE5B_BODY_ENVELOPE_RADIUS_M,
    launch_reference_height_m=_HOPPER_GATE5B_LAUNCH_REFERENCE_HEIGHT_M,
    arc_clearance_margin_m=_HOPPER_GATE5B_ARC_CLEARANCE_MARGIN_M,
    landing_footprint_radius_m=_HOPPER_GATE5B_LANDING_FOOTPRINT_RADIUS_M,
    stop_condition=_HOPPER_GATE5B_STOP_CONDITION_ID_V1,
    energy_model=_HOPPER_GATE5B_ENERGY_MODEL_ID_V1,
    stop_evaluator=_TRUSTED_HOPPER_GATE5B_STOP_EVALUATOR_V1,
    energy_evaluator=_TRUSTED_HOPPER_GATE5B_ENERGY_EVALUATOR_V1,
    evidence_class="test_fixture",
    simulation_proxy=True,
    formal_evidence_eligible=False,
    schema_version=_HOPPER_PARAMETER_SET_SCHEMA_VERSION_V2,
)

HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_IMPLEMENTATION_V1 = (
    HopperGenericInternalSimulationProxyImplementationRecordV2(
        parameter_set_id=(
            HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_PARAMETER_SET_ID_V1
        ),
        base_profile_id=_HOPPER_GENERIC_INTERNAL_BASE_PROFILE_ID_V1,
        capability_revision=(
            HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2
        ),
        body_envelope_radius_m=(
            _HOPPER_GENERIC_INTERNAL_BODY_ENVELOPE_RADIUS_M
        ),
        launch_reference_height_m=(
            _HOPPER_GENERIC_INTERNAL_LAUNCH_REFERENCE_HEIGHT_M
        ),
        arc_clearance_margin_m=(
            _HOPPER_GENERIC_INTERNAL_ARC_CLEARANCE_MARGIN_M
        ),
        landing_footprint_radius_m=(
            _HOPPER_GENERIC_INTERNAL_LANDING_FOOTPRINT_RADIUS_M
        ),
        stop_condition=_HOPPER_GENERIC_INTERNAL_STOP_CONDITION_ID_V1,
        stop_evaluator_id=_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_ID_V1,
        stop_evaluator_source_sha256=(
            _HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_SOURCE_SHA256_V1
        ),
        energy_model=_HOPPER_GENERIC_INTERNAL_ENERGY_MODEL_ID_V1,
        energy_evaluator_id=_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_ID_V1,
        energy_evaluator_source_sha256=(
            _HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_SOURCE_SHA256_V1
        ),
        stop_evaluator=(
            _TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_V1
        ),
        energy_evaluator=(
            _TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_V1
        ),
        evidence_class=_HOPPER_GENERIC_INTERNAL_EVIDENCE_CLASS_V1,
        simulation_proxy=True,
        physical_capability_claimed=False,
        hardware_certification_claimed=False,
        formal_evidence_eligible=False,
        schema_version=(
            _HOPPER_GENERIC_INTERNAL_IMPLEMENTATION_SCHEMA_VERSION_V1
        ),
    )
)

HOPPER_PARAMETER_SET_REGISTRY_V2 = (
    HOPPER_GATE5B_ALGORITHM_FIXTURE_V1,
    HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_IMPLEMENTATION_V1,
)


def _gate5b_parameter_set_record_is_exact_v2(record: object) -> bool:
    if type(record) is not HopperParameterSetRecordV2:
        return False
    try:
        return (
            type(record.parameter_set_id) is str
            and record.parameter_set_id == _HOPPER_GATE5B_PARAMETER_SET_ID_V1
            and type(record.base_profile_id) is str
            and record.base_profile_id == _HOPPER_GATE5B_BASE_PROFILE_ID_V1
            and type(record.body_envelope_radius_m) is float
            and record.body_envelope_radius_m.hex()
            == _HOPPER_GATE5B_BODY_ENVELOPE_RADIUS_M.hex()
            and type(record.launch_reference_height_m) is float
            and record.launch_reference_height_m.hex()
            == _HOPPER_GATE5B_LAUNCH_REFERENCE_HEIGHT_M.hex()
            and type(record.arc_clearance_margin_m) is float
            and record.arc_clearance_margin_m.hex()
            == _HOPPER_GATE5B_ARC_CLEARANCE_MARGIN_M.hex()
            and type(record.landing_footprint_radius_m) is float
            and record.landing_footprint_radius_m.hex()
            == _HOPPER_GATE5B_LANDING_FOOTPRINT_RADIUS_M.hex()
            and type(record.stop_condition) is str
            and record.stop_condition == _HOPPER_GATE5B_STOP_CONDITION_ID_V1
            and type(record.energy_model) is str
            and record.energy_model == _HOPPER_GATE5B_ENERGY_MODEL_ID_V1
            and record.stop_evaluator
            is _TRUSTED_HOPPER_GATE5B_STOP_EVALUATOR_V1
            and record.energy_evaluator
            is _TRUSTED_HOPPER_GATE5B_ENERGY_EVALUATOR_V1
            and type(record.evidence_class) is str
            and record.evidence_class == "test_fixture"
            and type(record.simulation_proxy) is bool
            and record.simulation_proxy is True
            and type(record.formal_evidence_eligible) is bool
            and record.formal_evidence_eligible is False
            and type(record.schema_version) is str
            and record.schema_version == _HOPPER_PARAMETER_SET_SCHEMA_VERSION_V2
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _generic_internal_parameter_set_record_is_exact_v2(record: object) -> bool:
    if (
        type(record)
        is not HopperGenericInternalSimulationProxyImplementationRecordV2
    ):
        return False
    try:
        return (
            type(record.parameter_set_id) is str
            and record.parameter_set_id
            == HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_PARAMETER_SET_ID_V1
            and type(record.base_profile_id) is str
            and record.base_profile_id == _HOPPER_GENERIC_INTERNAL_BASE_PROFILE_ID_V1
            and type(record.capability_revision) is str
            and record.capability_revision
            == HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2
            and type(record.body_envelope_radius_m) is float
            and record.body_envelope_radius_m.hex()
            == _HOPPER_GENERIC_INTERNAL_BODY_ENVELOPE_RADIUS_M.hex()
            and type(record.launch_reference_height_m) is float
            and record.launch_reference_height_m.hex()
            == _HOPPER_GENERIC_INTERNAL_LAUNCH_REFERENCE_HEIGHT_M.hex()
            and type(record.arc_clearance_margin_m) is float
            and record.arc_clearance_margin_m.hex()
            == _HOPPER_GENERIC_INTERNAL_ARC_CLEARANCE_MARGIN_M.hex()
            and type(record.landing_footprint_radius_m) is float
            and record.landing_footprint_radius_m.hex()
            == _HOPPER_GENERIC_INTERNAL_LANDING_FOOTPRINT_RADIUS_M.hex()
            and type(record.stop_condition) is str
            and record.stop_condition
            == _HOPPER_GENERIC_INTERNAL_STOP_CONDITION_ID_V1
            and type(record.stop_evaluator_id) is str
            and record.stop_evaluator_id
            == _HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_ID_V1
            and type(record.stop_evaluator_source_sha256) is str
            and record.stop_evaluator_source_sha256
            == _HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_SOURCE_SHA256_V1
            and type(record.energy_model) is str
            and record.energy_model == _HOPPER_GENERIC_INTERNAL_ENERGY_MODEL_ID_V1
            and type(record.energy_evaluator_id) is str
            and record.energy_evaluator_id
            == _HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_ID_V1
            and type(record.energy_evaluator_source_sha256) is str
            and record.energy_evaluator_source_sha256
            == _HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_SOURCE_SHA256_V1
            and record.stop_evaluator
            is _TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_V1
            and record.stop_evaluator.__code__
            is _TRUSTED_HOPPER_GENERIC_INTERNAL_STOP_EVALUATOR_CODE_V1
            and record.energy_evaluator
            is _TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_V1
            and record.energy_evaluator.__code__
            is _TRUSTED_HOPPER_GENERIC_INTERNAL_ENERGY_EVALUATOR_CODE_V1
            and type(record.evidence_class) is str
            and record.evidence_class == _HOPPER_GENERIC_INTERNAL_EVIDENCE_CLASS_V1
            and type(record.simulation_proxy) is bool
            and record.simulation_proxy is True
            and type(record.physical_capability_claimed) is bool
            and record.physical_capability_claimed is False
            and type(record.hardware_certification_claimed) is bool
            and record.hardware_certification_claimed is False
            and type(record.formal_evidence_eligible) is bool
            and record.formal_evidence_eligible is False
            and type(record.schema_version) is str
            and record.schema_version
            == _HOPPER_GENERIC_INTERNAL_IMPLEMENTATION_SCHEMA_VERSION_V1
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _parameter_set_record_is_exact_v2(record: object) -> bool:
    return (
        _gate5b_parameter_set_record_is_exact_v2(record)
        or _generic_internal_parameter_set_record_is_exact_v2(record)
    )


def _hopper_profile_matches_parameter_set_record_v2(
    profile: object,
    record: object,
) -> bool:
    if type(profile) is not HopperProfileV2 or not _parameter_set_record_is_exact_v2(
        record
    ):
        return False
    try:
        audit = audit_hopper_profile_v2(profile)
    except (AttributeError, TypeError, ValueError):
        return False
    if not audit.complete:
        return False
    expected_revision = (
        HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2
        if type(record) is HopperParameterSetRecordV2
        else HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2
    )
    try:
        platform = profile.profile
        return (
            type(platform) is PlatformProfileV2
            and type(platform.profile_id) is str
            and platform.profile_id == record.base_profile_id
            and type(platform.capability_revision) is str
            and platform.capability_revision == expected_revision
            and platform.platform_kind is PlatformKindV2.HOPPER
            and type(platform.simulation_proxy) is bool
            and platform.simulation_proxy is True
            and type(platform.max_traversable_slope_deg) is float
            and platform.max_traversable_slope_deg.hex() == 30.0.hex()
            and type(profile.body_envelope_radius_m) is float
            and profile.body_envelope_radius_m.hex()
            == record.body_envelope_radius_m.hex()
            and type(profile.launch_reference_height_m) is float
            and profile.launch_reference_height_m.hex()
            == record.launch_reference_height_m.hex()
            and type(profile.arc_clearance_margin_m) is float
            and profile.arc_clearance_margin_m.hex()
            == record.arc_clearance_margin_m.hex()
            and type(profile.landing_footprint_radius_m) is float
            and profile.landing_footprint_radius_m.hex()
            == record.landing_footprint_radius_m.hex()
            and type(profile.stop_condition) is str
            and profile.stop_condition == record.stop_condition
            and type(profile.energy_model) is str
            and profile.energy_model == record.energy_model
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _lookup_hopper_parameter_set_v2(
    parameter_set_id: str | None,
) -> (
    HopperParameterSetRecordV2
    | HopperGenericInternalSimulationProxyImplementationRecordV2
    | tuple[str, str | None]
):
    if parameter_set_id is not None and type(parameter_set_id) is not str:
        raise TypeError("parameter_set_id must be exact str or None")
    _require_canonical_hopper_parameter_registry_v2()
    for record in HOPPER_PARAMETER_SET_REGISTRY_V2:
        if record.parameter_set_id == parameter_set_id:
            return record
    return (_HOPPER_PARAMETER_SET_ABSENT_SCHEMA_VERSION_V2, parameter_set_id)


def _hopper_parameter_set_in_memory_token_v2(
    record: (
        HopperParameterSetRecordV2
        | HopperGenericInternalSimulationProxyImplementationRecordV2
    ),
) -> tuple[object, ...]:
    if not _parameter_set_record_is_exact_v2(record):
        raise ValueError("hopper_authority_contract_mismatch")
    if type(record) is HopperParameterSetRecordV2:
        return (
            record.parameter_set_id,
            record.base_profile_id,
            record.body_envelope_radius_m.hex(),
            record.launch_reference_height_m.hex(),
            record.arc_clearance_margin_m.hex(),
            record.landing_footprint_radius_m.hex(),
            record.stop_condition,
            record.energy_model,
            record.stop_evaluator,
            record.energy_evaluator,
            record.evidence_class,
            record.simulation_proxy,
            record.formal_evidence_eligible,
            record.schema_version,
        )
    return (
        record.parameter_set_id,
        record.base_profile_id,
        record.capability_revision,
        record.body_envelope_radius_m.hex(),
        record.launch_reference_height_m.hex(),
        record.arc_clearance_margin_m.hex(),
        record.landing_footprint_radius_m.hex(),
        record.stop_condition,
        record.stop_evaluator_id,
        record.stop_evaluator_source_sha256,
        record.stop_evaluator,
        record.energy_model,
        record.energy_evaluator_id,
        record.energy_evaluator_source_sha256,
        record.energy_evaluator,
        record.evidence_class,
        record.simulation_proxy,
        record.physical_capability_claimed,
        record.hardware_certification_claimed,
        record.formal_evidence_eligible,
        record.schema_version,
    )


def _hopper_parameter_set_lineage_token_v2(
    record: (
        HopperParameterSetRecordV2
        | HopperGenericInternalSimulationProxyImplementationRecordV2
    ),
) -> tuple[object, ...]:
    in_memory = _hopper_parameter_set_in_memory_token_v2(record)
    if type(record) is HopperParameterSetRecordV2:
        return (*in_memory[:8], *in_memory[10:])
    return (*in_memory[:10], *in_memory[11:14], *in_memory[15:])


HOPPER_PARAMETER_SET_REGISTRY_IN_MEMORY_TOKEN_V2 = tuple(
    _hopper_parameter_set_in_memory_token_v2(record)
    for record in HOPPER_PARAMETER_SET_REGISTRY_V2
)
HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2 = tuple(
    _hopper_parameter_set_lineage_token_v2(record)
    for record in HOPPER_PARAMETER_SET_REGISTRY_V2
)
_CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_V2 = HOPPER_PARAMETER_SET_REGISTRY_V2
_CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_IN_MEMORY_TOKEN_V2 = (
    HOPPER_PARAMETER_SET_REGISTRY_IN_MEMORY_TOKEN_V2
)
_CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2 = (
    HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2
)


def _require_canonical_hopper_parameter_registry_v2() -> None:
    registry = HOPPER_PARAMETER_SET_REGISTRY_V2
    if (
        registry is not _CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_V2
        or type(registry) is not tuple
        or registry
        != (
            HOPPER_GATE5B_ALGORITHM_FIXTURE_V1,
            HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_IMPLEMENTATION_V1,
        )
    ):
        raise ValueError("hopper_authority_contract_mismatch")
    ids = tuple(record.parameter_set_id for record in registry)
    if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
        raise ValueError("hopper_authority_contract_mismatch")
    current_in_memory = tuple(
        _hopper_parameter_set_in_memory_token_v2(record) for record in registry
    )
    current_lineage = tuple(
        _hopper_parameter_set_lineage_token_v2(record) for record in registry
    )
    if (
        HOPPER_PARAMETER_SET_REGISTRY_IN_MEMORY_TOKEN_V2
        is not _CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_IN_MEMORY_TOKEN_V2
        or HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2
        is not _CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2
        or current_in_memory
        != _CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_IN_MEMORY_TOKEN_V2
        or current_lineage
        != _CANONICAL_HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2
    ):
        raise ValueError("hopper_authority_contract_mismatch")
