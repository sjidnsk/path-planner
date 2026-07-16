from __future__ import annotations

from dataclasses import dataclass, field

from path_planner.v2.contracts import ValidationEvidenceV2, ValidationLevelV2
from path_planner.v2.serialization import canonical_json_bytes


VALIDATION_CACHE_SCHEMA_VERSION_V2 = "path-planner-v2-validation-cache/v1"

_MISSING = object()
_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


@dataclass(frozen=True, slots=True, init=False)
class ValidationCacheKeyV2:
    schema_version: str
    platform_profile_hash: str
    terrain_snapshot_hash: str
    primitive_hash: str
    validation_level: ValidationLevelV2
    objective_profile_hash: str | None

    def __init__(
        self,
        schema_version: object = _MISSING,
        platform_profile_hash: object = _MISSING,
        terrain_snapshot_hash: object = _MISSING,
        primitive_hash: object = _MISSING,
        validation_level: object = _MISSING,
        objective_profile_hash: object = None,
    ) -> None:
        values = (
            schema_version,
            platform_profile_hash,
            terrain_snapshot_hash,
            primitive_hash,
            validation_level,
            objective_profile_hash,
        )
        if not _complete_key_values_are_valid(*values):
            raise ValueError("complete cache key is required")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "platform_profile_hash", platform_profile_hash)
        object.__setattr__(self, "terrain_snapshot_hash", terrain_snapshot_hash)
        object.__setattr__(self, "primitive_hash", primitive_hash)
        object.__setattr__(self, "validation_level", validation_level)
        object.__setattr__(self, "objective_profile_hash", objective_profile_hash)


def _complete_key_values_are_valid(
    schema_version: object,
    platform_profile_hash: object,
    terrain_snapshot_hash: object,
    primitive_hash: object,
    validation_level: object,
    objective_profile_hash: object,
) -> bool:
    return (
        type(schema_version) is str
        and schema_version == VALIDATION_CACHE_SCHEMA_VERSION_V2
        and _is_sha256_digest(platform_profile_hash)
        and _is_sha256_digest(terrain_snapshot_hash)
        and _is_sha256_digest(primitive_hash)
        and type(validation_level) is ValidationLevelV2
        and (
            objective_profile_hash is None
            or _is_sha256_digest(objective_profile_hash)
        )
    )


def _canonical_key_bytes(key: object) -> bytes | None:
    if type(key) is not ValidationCacheKeyV2:
        return None
    try:
        values = (
            key.schema_version,
            key.platform_profile_hash,
            key.terrain_snapshot_hash,
            key.primitive_hash,
            key.validation_level,
            key.objective_profile_hash,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if not _complete_key_values_are_valid(*values):
        return None
    return canonical_json_bytes(
        {
            "objective_profile_hash": key.objective_profile_hash,
            "platform_profile_hash": key.platform_profile_hash,
            "primitive_hash": key.primitive_hash,
            "schema_version": key.schema_version,
            "terrain_snapshot_hash": key.terrain_snapshot_hash,
            "validation_level": key.validation_level.value,
        }
    )


def _copy_evidence(evidence: object) -> ValidationEvidenceV2:
    if type(evidence) is not ValidationEvidenceV2:
        raise TypeError("evidence must be exact ValidationEvidenceV2")
    try:
        validator_id = evidence.validator_id
        level = evidence.level
        passed = evidence.passed
        checks = evidence.checks
    except (AttributeError, TypeError, ValueError):
        raise TypeError("evidence must be exact ValidationEvidenceV2") from None
    if (
        type(validator_id) is not str
        or not validator_id.strip()
        or type(level) is not ValidationLevelV2
        or type(passed) is not bool
        or type(checks) is not tuple
        or not checks
        or any(type(check) is not str or not check.strip() for check in checks)
    ):
        raise TypeError("evidence must be exact ValidationEvidenceV2")
    return ValidationEvidenceV2(
        validator_id=validator_id,
        level=level,
        passed=passed,
        checks=checks,
    )


def _is_transient(evidence: ValidationEvidenceV2) -> bool:
    return any(
        check == "planning_deadline_expired"
        or check.endswith("_contract_mismatch")
        for check in evidence.checks
    )


@dataclass(slots=True)
class ValidationCacheV2:
    _entries: dict[bytes, ValidationEvidenceV2] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _hit_count: int = field(default=0, init=False, repr=False)
    _miss_count: int = field(default=0, init=False, repr=False)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get(self, key: object) -> ValidationEvidenceV2 | None:
        canonical = _canonical_key_bytes(key)
        if canonical is None:
            self._miss_count += 1
            return None
        stored = self._entries.get(canonical)
        if stored is None:
            self._miss_count += 1
            return None
        try:
            copied = _copy_evidence(stored)
        except (TypeError, ValueError):
            self._miss_count += 1
            return None
        self._hit_count += 1
        return copied

    def put(self, key: object, evidence: object) -> None:
        canonical = _canonical_key_bytes(key)
        if canonical is None:
            raise ValueError("complete cache key is required")
        copied = _copy_evidence(evidence)
        if copied.level is not key.validation_level:
            raise ValueError("evidence level must match complete cache key level")
        if _is_transient(copied):
            raise ValueError("transient validation evidence is not cacheable")
        existing = self._entries.get(canonical)
        if existing is None:
            self._entries[canonical] = copied
            return
        if existing != copied:
            raise ValueError("conflicting validation evidence for complete cache key")
