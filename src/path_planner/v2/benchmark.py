from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
import re
from typing import Any, ClassVar, TypeVar


HARD_TIMEOUT_MS_V2 = 2_000.0

_STABLE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z", re.ASCII)


def _stable_token(value: object, name: str) -> None:
    if not isinstance(value, str) or _STABLE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")


def _strict_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


def _seed(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer and must not be bool")
    if value < 0:
        raise ValueError("seed must be nonnegative")


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_float(value: object, name: str) -> float:
    normalized = _nonnegative_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _strict_payload(
    payload: object,
    *,
    expected_keys: tuple[str, ...],
    row_name: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{row_name} payload must be a mapping")
    actual_keys = tuple(payload.keys())
    if any(not isinstance(key, str) for key in actual_keys) or set(actual_keys) != set(
        expected_keys
    ):
        raise ValueError(f"{row_name} payload must contain exact keys {expected_keys!r}")
    return payload


@dataclass(frozen=True, slots=True)
class PrimitiveAuditRowV2:
    row_id: str
    seed: int
    expected_safe: bool
    expected_label_source: str
    expected_label_independent: bool
    provider_safe: bool
    provider_complete_l2: bool
    runtime_ms: float
    timed_out: bool
    reason_code: str

    _KEYS: ClassVar[tuple[str, ...]] = (
        "row_id",
        "seed",
        "expected_safe",
        "expected_label_source",
        "expected_label_independent",
        "provider_safe",
        "provider_complete_l2",
        "runtime_ms",
        "timed_out",
        "reason_code",
    )

    def __post_init__(self) -> None:
        _stable_token(self.row_id, "row_id")
        _seed(self.seed)
        for name in (
            "expected_safe",
            "expected_label_independent",
            "provider_safe",
            "provider_complete_l2",
            "timed_out",
        ):
            _strict_bool(getattr(self, name), name)
        _stable_token(self.expected_label_source, "expected_label_source")
        object.__setattr__(self, "runtime_ms", _nonnegative_float(self.runtime_ms, "runtime_ms"))
        _stable_token(self.reason_code, "reason_code")
        if self.provider_complete_l2 and not self.provider_safe:
            raise ValueError("provider complete L2 requires provider_safe")
        if self.timed_out and (self.provider_safe or self.provider_complete_l2):
            raise ValueError("timed out row cannot report provider success")

    @classmethod
    def from_dict(cls, payload: object) -> PrimitiveAuditRowV2:
        values = _strict_payload(payload, expected_keys=cls._KEYS, row_name=cls.__name__)
        return cls(**{key: values[key] for key in cls._KEYS})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True, slots=True)
class ExactMapQualityRowV2:
    row_id: str
    seed: int
    optimum_resource_cost: float
    optimum_source: str
    optimum_independent: bool
    provider_success: bool
    provider_resource_cost: float | None
    provider_complete_l2: bool
    runtime_ms: float
    timed_out: bool
    reason_code: str

    _KEYS: ClassVar[tuple[str, ...]] = (
        "row_id",
        "seed",
        "optimum_resource_cost",
        "optimum_source",
        "optimum_independent",
        "provider_success",
        "provider_resource_cost",
        "provider_complete_l2",
        "runtime_ms",
        "timed_out",
        "reason_code",
    )

    def __post_init__(self) -> None:
        _stable_token(self.row_id, "row_id")
        _seed(self.seed)
        object.__setattr__(
            self,
            "optimum_resource_cost",
            _positive_float(self.optimum_resource_cost, "optimum_resource_cost"),
        )
        _stable_token(self.optimum_source, "optimum_source")
        for name in (
            "optimum_independent",
            "provider_success",
            "provider_complete_l2",
            "timed_out",
        ):
            _strict_bool(getattr(self, name), name)
        if self.provider_resource_cost is not None:
            object.__setattr__(
                self,
                "provider_resource_cost",
                _nonnegative_float(self.provider_resource_cost, "provider_resource_cost"),
            )
        object.__setattr__(self, "runtime_ms", _nonnegative_float(self.runtime_ms, "runtime_ms"))
        _stable_token(self.reason_code, "reason_code")
        if self.provider_success != (self.provider_resource_cost is not None):
            raise ValueError(
                "provider resource cost must be present exactly when provider succeeds"
            )
        if self.provider_complete_l2 and not self.provider_success:
            raise ValueError("provider complete L2 requires provider success")
        if self.timed_out and (self.provider_success or self.provider_complete_l2):
            raise ValueError("timed out row cannot report provider success")

    @classmethod
    def from_dict(cls, payload: object) -> ExactMapQualityRowV2:
        values = _strict_payload(payload, expected_keys=cls._KEYS, row_name=cls.__name__)
        return cls(**{key: values[key] for key in cls._KEYS})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True, slots=True)
class StandardEpisodeRowV2:
    episode_id: str
    seed: int
    schedule_source: str
    schedule_independent: bool
    oracle_reachable: bool
    provider_success: bool
    provider_complete_l2: bool
    runtime_ms: float
    timed_out: bool
    reason_code: str

    _KEYS: ClassVar[tuple[str, ...]] = (
        "episode_id",
        "seed",
        "schedule_source",
        "schedule_independent",
        "oracle_reachable",
        "provider_success",
        "provider_complete_l2",
        "runtime_ms",
        "timed_out",
        "reason_code",
    )

    def __post_init__(self) -> None:
        _stable_token(self.episode_id, "episode_id")
        _seed(self.seed)
        _stable_token(self.schedule_source, "schedule_source")
        for name in (
            "schedule_independent",
            "oracle_reachable",
            "provider_success",
            "provider_complete_l2",
            "timed_out",
        ):
            _strict_bool(getattr(self, name), name)
        object.__setattr__(self, "runtime_ms", _nonnegative_float(self.runtime_ms, "runtime_ms"))
        _stable_token(self.reason_code, "reason_code")
        if self.provider_complete_l2 and not self.provider_success:
            raise ValueError("provider complete L2 requires provider success")
        if self.timed_out and (self.provider_success or self.provider_complete_l2):
            raise ValueError("timed out row cannot report provider success")

    @classmethod
    def from_dict(cls, payload: object) -> StandardEpisodeRowV2:
        values = _strict_payload(payload, expected_keys=cls._KEYS, row_name=cls.__name__)
        return cls(**{key: values[key] for key in cls._KEYS})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True, slots=True)
class PrimitiveAuditSummaryV2:
    rows: tuple[PrimitiveAuditRowV2, ...]
    total_rows: int
    formal_row_count: int
    excluded_row_count: int
    false_positive_count: int
    true_positive_count: int
    false_negative_count: int
    primitive_recall: float | None
    provider_success_count: int
    complete_l2_ratio: float | None
    runtime_p50_ms: float | None
    runtime_p95_ms: float | None
    runtime_p99_ms: float | None
    timeout_count: int
    hard_timeout_violation_count: int
    reason_histogram: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "formal_row_count": self.formal_row_count,
            "excluded_row_count": self.excluded_row_count,
            "false_positive_count": self.false_positive_count,
            "true_positive_count": self.true_positive_count,
            "false_negative_count": self.false_negative_count,
            "primitive_recall": self.primitive_recall,
            "provider_success_count": self.provider_success_count,
            "complete_l2_ratio": self.complete_l2_ratio,
            "runtime_p50_ms": self.runtime_p50_ms,
            "runtime_p95_ms": self.runtime_p95_ms,
            "runtime_p99_ms": self.runtime_p99_ms,
            "timeout_count": self.timeout_count,
            "hard_timeout_violation_count": self.hard_timeout_violation_count,
            "reason_histogram": dict(self.reason_histogram),
        }


@dataclass(frozen=True, slots=True)
class ExactMapQualitySummaryV2:
    rows: tuple[ExactMapQualityRowV2, ...]
    total_rows: int
    formal_row_count: int
    excluded_row_count: int
    provider_success_count: int
    provider_success_ratio: float | None
    complete_l2_ratio: float | None
    resource_cost_ratios: tuple[tuple[str, float], ...]
    max_resource_cost_ratio: float | None
    runtime_p50_ms: float | None
    runtime_p95_ms: float | None
    runtime_p99_ms: float | None
    timeout_count: int
    hard_timeout_violation_count: int
    reason_histogram: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "formal_row_count": self.formal_row_count,
            "excluded_row_count": self.excluded_row_count,
            "provider_success_count": self.provider_success_count,
            "provider_success_ratio": self.provider_success_ratio,
            "complete_l2_ratio": self.complete_l2_ratio,
            "resource_cost_ratios": [
                {"row_id": row_id, "ratio": ratio}
                for row_id, ratio in self.resource_cost_ratios
            ],
            "max_resource_cost_ratio": self.max_resource_cost_ratio,
            "runtime_p50_ms": self.runtime_p50_ms,
            "runtime_p95_ms": self.runtime_p95_ms,
            "runtime_p99_ms": self.runtime_p99_ms,
            "timeout_count": self.timeout_count,
            "hard_timeout_violation_count": self.hard_timeout_violation_count,
            "reason_histogram": dict(self.reason_histogram),
        }


@dataclass(frozen=True, slots=True)
class StandardEpisodeSummaryV2:
    rows: tuple[StandardEpisodeRowV2, ...]
    total_episodes: int
    formal_episode_count: int
    excluded_episode_count: int
    formal_oracle_reachable_count: int
    reachable_provider_success_count: int
    reachable_query_success_ratio: float | None
    complete_l2_ratio: float | None
    runtime_p50_ms: float | None
    runtime_p95_ms: float | None
    runtime_p99_ms: float | None
    timeout_count: int
    hard_timeout_violation_count: int
    reason_histogram: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_episodes": self.total_episodes,
            "formal_episode_count": self.formal_episode_count,
            "excluded_episode_count": self.excluded_episode_count,
            "formal_oracle_reachable_count": self.formal_oracle_reachable_count,
            "reachable_provider_success_count": self.reachable_provider_success_count,
            "reachable_query_success_ratio": self.reachable_query_success_ratio,
            "complete_l2_ratio": self.complete_l2_ratio,
            "runtime_p50_ms": self.runtime_p50_ms,
            "runtime_p95_ms": self.runtime_p95_ms,
            "runtime_p99_ms": self.runtime_p99_ms,
            "timeout_count": self.timeout_count,
            "hard_timeout_violation_count": self.hard_timeout_violation_count,
            "reason_histogram": dict(self.reason_histogram),
        }


_RowT = TypeVar("_RowT")


def _sorted_unique_rows(
    rows: Iterable[_RowT],
    *,
    row_type: type[_RowT],
    id_attribute: str,
) -> tuple[_RowT, ...]:
    materialized = tuple(rows)
    if any(not isinstance(row, row_type) for row in materialized):
        raise TypeError(f"rows must contain only {row_type.__name__} values")
    ids = [getattr(row, id_attribute) for row in materialized]
    duplicate_ids = sorted(row_id for row_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate stable ids: {', '.join(duplicate_ids)}")
    return tuple(
        sorted(
            materialized,
            key=lambda row: (getattr(row, "seed"), getattr(row, id_attribute)),
        )
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _nearest_rank(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[ceil(quantile * len(ordered)) - 1]


def _runtime_metrics(rows: Iterable[Any]) -> tuple[float | None, float | None, float | None]:
    runtimes = tuple(row.runtime_ms for row in rows)
    return (
        _nearest_rank(runtimes, 0.50),
        _nearest_rank(runtimes, 0.95),
        _nearest_rank(runtimes, 0.99),
    )


def _reason_histogram(rows: Iterable[Any]) -> tuple[tuple[str, int], ...]:
    counts = Counter(row.reason_code for row in rows)
    return tuple(sorted(counts.items()))


def _resource_cost_ratios(
    rows: Iterable[ExactMapQualityRowV2],
) -> tuple[tuple[str, float], ...]:
    ratios: list[tuple[str, float]] = []
    for row in rows:
        if row.provider_resource_cost is None:
            continue
        ratio = row.provider_resource_cost / row.optimum_resource_cost
        if not isfinite(ratio):
            raise ValueError(f"resource cost ratio for {row.row_id} must be finite")
        ratios.append((row.row_id, ratio))
    return tuple(ratios)


def aggregate_primitive_audit_v2(
    rows: Iterable[PrimitiveAuditRowV2],
) -> PrimitiveAuditSummaryV2:
    ordered = _sorted_unique_rows(
        rows,
        row_type=PrimitiveAuditRowV2,
        id_attribute="row_id",
    )
    formal = tuple(row for row in ordered if row.expected_label_independent)
    false_positives = sum(row.provider_safe and not row.expected_safe for row in formal)
    true_positives = sum(row.provider_safe and row.expected_safe for row in formal)
    false_negatives = sum(not row.provider_safe and row.expected_safe for row in formal)
    provider_successes = sum(row.provider_safe for row in formal)
    complete_l2 = sum(row.provider_safe and row.provider_complete_l2 for row in formal)
    p50, p95, p99 = _runtime_metrics(formal)
    return PrimitiveAuditSummaryV2(
        rows=ordered,
        total_rows=len(ordered),
        formal_row_count=len(formal),
        excluded_row_count=len(ordered) - len(formal),
        false_positive_count=false_positives,
        true_positive_count=true_positives,
        false_negative_count=false_negatives,
        primitive_recall=_ratio(true_positives, true_positives + false_negatives),
        provider_success_count=provider_successes,
        complete_l2_ratio=_ratio(complete_l2, provider_successes),
        runtime_p50_ms=p50,
        runtime_p95_ms=p95,
        runtime_p99_ms=p99,
        timeout_count=sum(row.timed_out for row in formal),
        hard_timeout_violation_count=sum(row.runtime_ms > HARD_TIMEOUT_MS_V2 for row in formal),
        reason_histogram=_reason_histogram(formal),
    )


def aggregate_exact_map_quality_v2(
    rows: Iterable[ExactMapQualityRowV2],
) -> ExactMapQualitySummaryV2:
    ordered = _sorted_unique_rows(
        rows,
        row_type=ExactMapQualityRowV2,
        id_attribute="row_id",
    )
    formal = tuple(row for row in ordered if row.optimum_independent)
    successful = tuple(row for row in formal if row.provider_success)
    ratios = _resource_cost_ratios(successful)
    p50, p95, p99 = _runtime_metrics(formal)
    return ExactMapQualitySummaryV2(
        rows=ordered,
        total_rows=len(ordered),
        formal_row_count=len(formal),
        excluded_row_count=len(ordered) - len(formal),
        provider_success_count=len(successful),
        provider_success_ratio=_ratio(len(successful), len(formal)),
        complete_l2_ratio=_ratio(
            sum(row.provider_complete_l2 for row in successful),
            len(successful),
        ),
        resource_cost_ratios=ratios,
        max_resource_cost_ratio=max((ratio for _, ratio in ratios), default=None),
        runtime_p50_ms=p50,
        runtime_p95_ms=p95,
        runtime_p99_ms=p99,
        timeout_count=sum(row.timed_out for row in formal),
        hard_timeout_violation_count=sum(row.runtime_ms > HARD_TIMEOUT_MS_V2 for row in formal),
        reason_histogram=_reason_histogram(formal),
    )


def aggregate_standard_episodes_v2(
    rows: Iterable[StandardEpisodeRowV2],
) -> StandardEpisodeSummaryV2:
    ordered = _sorted_unique_rows(
        rows,
        row_type=StandardEpisodeRowV2,
        id_attribute="episode_id",
    )
    formal = tuple(row for row in ordered if row.schedule_independent)
    reachable = tuple(row for row in formal if row.oracle_reachable)
    reachable_successes = sum(row.provider_success for row in reachable)
    provider_successes = tuple(row for row in formal if row.provider_success)
    p50, p95, p99 = _runtime_metrics(formal)
    return StandardEpisodeSummaryV2(
        rows=ordered,
        total_episodes=len(ordered),
        formal_episode_count=len(formal),
        excluded_episode_count=len(ordered) - len(formal),
        formal_oracle_reachable_count=len(reachable),
        reachable_provider_success_count=reachable_successes,
        reachable_query_success_ratio=_ratio(reachable_successes, len(reachable)),
        complete_l2_ratio=_ratio(
            sum(row.provider_complete_l2 for row in provider_successes),
            len(provider_successes),
        ),
        runtime_p50_ms=p50,
        runtime_p95_ms=p95,
        runtime_p99_ms=p99,
        timeout_count=sum(row.timed_out for row in formal),
        hard_timeout_violation_count=sum(row.runtime_ms > HARD_TIMEOUT_MS_V2 for row in formal),
        reason_histogram=_reason_histogram(formal),
    )


__all__ = [
    "HARD_TIMEOUT_MS_V2",
    "ExactMapQualityRowV2",
    "ExactMapQualitySummaryV2",
    "PrimitiveAuditRowV2",
    "PrimitiveAuditSummaryV2",
    "StandardEpisodeRowV2",
    "StandardEpisodeSummaryV2",
    "aggregate_exact_map_quality_v2",
    "aggregate_primitive_audit_v2",
    "aggregate_standard_episodes_v2",
]
