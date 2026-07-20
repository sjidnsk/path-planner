from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import path_planner.v2.ballistics as _ballistics


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

_BALLISTIC_HELPER_ID = "sample_ballistic_arc_capped/v1"
_LANDING_HELPER_ID = "landing_zone_cells_capped/v1"
_MEMORY_ACCOUNTING_ID = "hopper_deterministic_admission_bytes/v1"
_RESOURCE_SCHEMA_VERSION = "hopper-resource-authority/v1"
_RESOURCE_LINEAGE_SCHEMA_VERSION = "hopper-resource-authority-lineage/v1"

_TRUSTED_BALLISTIC_HELPER = _ballistics.sample_ballistic_arc_capped_v2
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
        getattr(_ballistics, "sample_ballistic_arc_capped_v2", None)
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
    azimuth_rad: float,
    g_mps2: float,
    dt_s: float,
) -> tuple[_ballistics.BallisticSampleV2, ...]:
    _require_canonical_resource_authority_v2(authority)
    helper = authority.ballistic_helper
    cap = authority.max_ballistic_samples
    try:
        result = helper(
            start,
            speed_mps,
            elevation_rad,
            azimuth_rad,
            g_mps2,
            dt_s,
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
