from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, floor, isfinite
from numbers import Real
import random
import re
from typing import Any


GATE6_PLATFORMS_V2 = ("wheel", "legged", "hopper")
GATE6_PHASES_V2 = (
    "primitive_audit",
    "exact_quality",
    "standard",
    "kilometer",
    "ablation",
    "determinism_worker_cache",
    "aggregate",
)
GATE6_ABLATION_CASES_V2 = (
    "v1_astar",
    "wheel_hybrid_astar_opt_in",
    "v2_fine_only",
    "v2_plus_multi_heuristic",
    "v2_plus_hierarchy",
    "v2_plus_lazy_validation",
    "v2_plus_cache",
    "v2_full",
)
GATE6_BOOTSTRAP_SEED_V2 = 20260716
GATE6_BOOTSTRAP_RESAMPLES_V2 = 10_000
GATE6_HARD_TIMEOUT_MS_V2 = 2_000.0

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z", re.ASCII)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def _stable_token(value: object, field: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a stable identifier")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be bool")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _positive_int(value: object, field: str) -> int:
    normalized = _nonnegative_int(value, field)
    if normalized == 0:
        raise ValueError(f"{field} must be positive")
    return normalized


def _nonnegative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return normalized


def _optional_nonnegative_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_float(value, field)


@dataclass(frozen=True, slots=True)
class Gate6FixtureRequirementV2:
    input_kind: str
    blocker: str
    minimum_rows_per_platform: int
    requires_independent_source: bool = True

    def __post_init__(self) -> None:
        _stable_token(self.input_kind, "input_kind")
        _stable_token(self.blocker, "blocker")
        _positive_int(self.minimum_rows_per_platform, "minimum_rows_per_platform")
        _strict_bool(self.requires_independent_source, "requires_independent_source")


GATE6_REQUIRED_INPUTS_V2 = (
    Gate6FixtureRequirementV2(
        "independent_primitive_labels",
        "provide_independent_10000_primitive_labels_per_platform",
        10_000,
    ),
    Gate6FixtureRequirementV2(
        "independent_small_map_optima",
        "provide_independent_small_map_optima",
        1,
    ),
    Gate6FixtureRequirementV2(
        "standard_schedules",
        "provide_standard_100_episodes_per_platform",
        100,
    ),
    Gate6FixtureRequirementV2(
        "kilometer_schedules",
        "provide_kilometer_30_episodes_per_platform",
        30,
    ),
    Gate6FixtureRequirementV2(
        "ppo_targets",
        "provide_ppo_target_fixtures",
        1,
    ),
)


@dataclass(frozen=True, slots=True)
class Gate6FixtureInventoryV2:
    input_kind: str
    source_id: str
    source_independent: bool
    rows_per_platform: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _stable_token(self.input_kind, "input_kind")
        _stable_token(self.source_id, "source_id")
        _strict_bool(self.source_independent, "source_independent")
        if type(self.rows_per_platform) is not tuple:
            raise TypeError("rows_per_platform must be a tuple")
        platforms: list[str] = []
        for item in self.rows_per_platform:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("rows_per_platform must contain (platform, count) tuples")
            platform, count = item
            if platform not in GATE6_PLATFORMS_V2:
                raise ValueError("rows_per_platform contains unknown platform")
            _nonnegative_int(count, "platform row count")
            platforms.append(platform)
        if tuple(platforms) != GATE6_PLATFORMS_V2:
            raise ValueError("rows_per_platform must use exact platform order")


@dataclass(frozen=True, slots=True)
class Gate6FixtureInputAuditV2:
    input_kind: str
    status: str
    blocker: str | None
    formal_row_count: int


@dataclass(frozen=True, slots=True)
class Gate6FixtureAuditV2:
    inputs: tuple[Gate6FixtureInputAuditV2, ...]
    blocking_reasons: tuple[str, ...]
    formal_evidence_eligible: bool
    formal_row_count: int


def audit_gate6_fixture_inventory_v2(
    inventories: Mapping[str, Gate6FixtureInventoryV2 | None],
) -> Gate6FixtureAuditV2:
    if not isinstance(inventories, Mapping):
        raise TypeError("inventories must be a mapping")
    expected = tuple(item.input_kind for item in GATE6_REQUIRED_INPUTS_V2)
    if set(inventories) != set(expected) or any(type(key) is not str for key in inventories):
        raise ValueError("inventories must contain the exact Gate 6 input kinds")

    audits: list[Gate6FixtureInputAuditV2] = []
    blockers: list[str] = []
    formal_row_count = 0
    for requirement in GATE6_REQUIRED_INPUTS_V2:
        inventory = inventories[requirement.input_kind]
        if inventory is None:
            status = "missing"
            count = 0
        elif type(inventory) is not Gate6FixtureInventoryV2:
            raise TypeError("inventory values must be Gate6FixtureInventoryV2 or None")
        elif inventory.input_kind != requirement.input_kind:
            raise ValueError("inventory input_kind does not match its mapping key")
        elif requirement.requires_independent_source and not inventory.source_independent:
            status = "non_independent"
            count = 0
        else:
            counts = dict(inventory.rows_per_platform)
            count = sum(counts.values())
            status = (
                "ready"
                if all(
                    counts[platform] >= requirement.minimum_rows_per_platform
                    for platform in GATE6_PLATFORMS_V2
                )
                else "insufficient"
            )
        blocker = None if status == "ready" else requirement.blocker
        if blocker is not None:
            blockers.append(blocker)
        else:
            formal_row_count += count
        audits.append(
            Gate6FixtureInputAuditV2(
                input_kind=requirement.input_kind,
                status=status,
                blocker=blocker,
                formal_row_count=count if status == "ready" else 0,
            )
        )
    return Gate6FixtureAuditV2(
        inputs=tuple(audits),
        blocking_reasons=tuple(blockers),
        formal_evidence_eligible=not blockers,
        formal_row_count=formal_row_count,
    )


@dataclass(frozen=True, slots=True)
class Gate6EpisodeMetricRowV2:
    episode_id: str
    pair_id: str
    platform_kind: str
    scale: str
    ablation_case: str
    seed: int
    worker_count: int
    cache_enabled: bool
    oracle_reachable: bool
    provider_success: bool
    provider_complete_l2: bool
    primitive_true_positive_count: int
    primitive_false_negative_count: int
    primitive_false_positive_count: int
    resource_cost: float | None
    coverage_efficiency: float
    expanded_states: int
    rejected_l0: int
    rejected_l1: int
    rejected_l2: int
    cache_hits: int
    cache_lookups: int
    runtime_ms: float
    timed_out: bool
    semantic_digest: str

    def __post_init__(self) -> None:
        _stable_token(self.episode_id, "episode_id")
        _stable_token(self.pair_id, "pair_id")
        if self.platform_kind not in GATE6_PLATFORMS_V2:
            raise ValueError("platform_kind is not supported")
        if self.scale not in {"standard", "kilometer"}:
            raise ValueError("scale must be standard or kilometer")
        if self.ablation_case not in GATE6_ABLATION_CASES_V2:
            raise ValueError("ablation_case is not frozen")
        if self.ablation_case in {"v1_astar", "wheel_hybrid_astar_opt_in"} and (
            self.platform_kind != "wheel"
        ):
            raise ValueError("v1 and Hybrid A* baseline cases are wheel-only")
        _nonnegative_int(self.seed, "seed")
        _positive_int(self.worker_count, "worker_count")
        for field in (
            "cache_enabled",
            "oracle_reachable",
            "provider_success",
            "provider_complete_l2",
            "timed_out",
        ):
            _strict_bool(getattr(self, field), field)
        for field in (
            "primitive_true_positive_count",
            "primitive_false_negative_count",
            "primitive_false_positive_count",
            "expanded_states",
            "rejected_l0",
            "rejected_l1",
            "rejected_l2",
            "cache_hits",
            "cache_lookups",
        ):
            _nonnegative_int(getattr(self, field), field)
        if self.cache_hits > self.cache_lookups:
            raise ValueError("cache_hits cannot exceed cache_lookups")
        object.__setattr__(
            self,
            "resource_cost",
            _optional_nonnegative_float(self.resource_cost, "resource_cost"),
        )
        if self.coverage_efficiency is None:
            raise ValueError("coverage_efficiency is required for every episode")
        object.__setattr__(
            self,
            "coverage_efficiency",
            _nonnegative_float(self.coverage_efficiency, "coverage_efficiency"),
        )
        object.__setattr__(self, "runtime_ms", _nonnegative_float(self.runtime_ms, "runtime_ms"))
        if self.provider_complete_l2 and not self.provider_success:
            raise ValueError("complete L2 requires provider success")
        if self.timed_out and self.provider_success:
            raise ValueError("timed out row cannot report provider success")
        if self.provider_success != (self.resource_cost is not None):
            raise ValueError("resource_cost must be present exactly for successful rows")
        if type(self.semantic_digest) is not str or _LOWER_SHA256.fullmatch(
            self.semantic_digest
        ) is None:
            raise ValueError("semantic_digest must be lowercase sha256")


def _ordered_metric_rows(
    rows: Iterable[Gate6EpisodeMetricRowV2],
) -> tuple[Gate6EpisodeMetricRowV2, ...]:
    materialized = tuple(rows)
    if any(type(row) is not Gate6EpisodeMetricRowV2 for row in materialized):
        raise TypeError("rows must contain only Gate6EpisodeMetricRowV2")
    ids = [row.episode_id for row in materialized]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate episode ids: {', '.join(duplicates)}")
    return tuple(
        sorted(
            materialized,
            key=lambda row: (
                row.seed,
                row.platform_kind,
                row.scale,
                row.ablation_case,
                row.pair_id,
                row.worker_count,
                row.cache_enabled,
                row.episode_id,
            ),
        )
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Iterable[float]) -> float | None:
    materialized = tuple(values)
    return None if not materialized else sum(materialized) / len(materialized)


def _nearest_rank(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[ceil(quantile * len(ordered)) - 1]


def aggregate_gate6_metrics_v2(
    rows: Iterable[Gate6EpisodeMetricRowV2],
) -> dict[str, Any]:
    ordered = _ordered_metric_rows(rows)
    reachable = tuple(row for row in ordered if row.oracle_reachable)
    successful = tuple(row for row in ordered if row.provider_success)
    true_positives = sum(row.primitive_true_positive_count for row in ordered)
    false_negatives = sum(row.primitive_false_negative_count for row in ordered)
    false_positives = sum(row.primitive_false_positive_count for row in ordered)
    unsafe_successes = sum(
        row.provider_success and not row.provider_complete_l2 for row in ordered
    )
    cache_hits = sum(row.cache_hits for row in ordered)
    cache_lookups = sum(row.cache_lookups for row in ordered)
    runtimes = tuple(row.runtime_ms for row in ordered)
    return {
        "metrics_status": "evaluated" if ordered else "not_evaluated",
        "row_count": len(ordered),
        "safety_passed": (
            false_positives == 0 and unsafe_successes == 0 if ordered else None
        ),
        "unsafe_success_count": unsafe_successes,
        "primitive_false_positive_count": false_positives,
        "primitive_recall": _ratio(true_positives, true_positives + false_negatives),
        "oracle_reachable_count": len(reachable),
        "reachable_success_count": sum(row.provider_success for row in reachable),
        "reachable_success_ratio": _ratio(
            sum(row.provider_success for row in reachable),
            len(reachable),
        ),
        "mean_resource_cost": _mean(
            row.resource_cost for row in successful if row.resource_cost is not None
        ),
        "mean_coverage_efficiency": _mean(
            row.coverage_efficiency for row in ordered
        ),
        "expanded_states": sum(row.expanded_states for row in ordered),
        "rejected_l0": sum(row.rejected_l0 for row in ordered),
        "rejected_l1": sum(row.rejected_l1 for row in ordered),
        "rejected_l2": sum(row.rejected_l2 for row in ordered),
        "cache_hits": cache_hits,
        "cache_lookups": cache_lookups,
        "cache_hit_ratio": _ratio(cache_hits, cache_lookups),
        "runtime_p50_ms": _nearest_rank(runtimes, 0.50),
        "runtime_p95_ms": _nearest_rank(runtimes, 0.95),
        "runtime_p99_ms": _nearest_rank(runtimes, 0.99),
        "timeout_count": sum(row.timed_out for row in ordered),
        "timeout_ratio": _ratio(sum(row.timed_out for row in ordered), len(ordered)),
        "hard_timeout_violation_count": sum(
            row.runtime_ms > GATE6_HARD_TIMEOUT_MS_V2 for row in ordered
        ),
    }


def _paired_key(row: Gate6EpisodeMetricRowV2) -> tuple[object, ...]:
    return (
        row.platform_kind,
        row.scale,
        row.pair_id,
        row.seed,
    )


def paired_bootstrap_coverage_ci95_v2(
    rows: Iterable[Gate6EpisodeMetricRowV2],
    *,
    baseline_case: str,
    candidate_case: str,
    seed: int = GATE6_BOOTSTRAP_SEED_V2,
    resamples: int = GATE6_BOOTSTRAP_RESAMPLES_V2,
) -> dict[str, int | float | str]:
    if baseline_case not in GATE6_ABLATION_CASES_V2:
        raise ValueError("baseline_case is not frozen")
    if candidate_case not in GATE6_ABLATION_CASES_V2 or candidate_case == baseline_case:
        raise ValueError("candidate_case must be a distinct frozen case")
    _nonnegative_int(seed, "seed")
    _positive_int(resamples, "resamples")
    materialized = tuple(rows)
    if any(type(row) is not Gate6EpisodeMetricRowV2 for row in materialized):
        raise TypeError("rows must contain only Gate6EpisodeMetricRowV2")

    def by_key(case_id: str) -> dict[tuple[object, ...], float]:
        result: dict[tuple[object, ...], float] = {}
        for row in materialized:
            if (
                row.ablation_case != case_id
                or row.worker_count != 1
                or row.cache_enabled
            ):
                continue
            key = _paired_key(row)
            if key in result:
                raise ValueError("duplicate paired coverage key")
            result[key] = row.coverage_efficiency
        return result

    baseline = by_key(baseline_case)
    candidate = by_key(candidate_case)
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("paired coverage cases must contain identical nonempty keys")
    deltas = tuple(candidate[key] - baseline[key] for key in sorted(baseline))
    mean_delta = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(resamples):
        sample = tuple(deltas[rng.randrange(len(deltas))] for _ in deltas)
        bootstrap_means.append(sum(sample) / len(sample))
    ordered = sorted(bootstrap_means)
    lower_index = floor(0.025 * (len(ordered) - 1))
    upper_index = ceil(0.975 * (len(ordered) - 1))
    return {
        "baseline_case": baseline_case,
        "candidate_case": candidate_case,
        "pair_count": len(deltas),
        "mean_delta": mean_delta,
        "ci95_lower": ordered[lower_index],
        "ci95_upper": ordered[upper_index],
        "seed": seed,
        "resamples": resamples,
    }


def audit_gate6_worker_cache_semantics_v2(
    rows: Iterable[Gate6EpisodeMetricRowV2],
    *,
    worker_counts: tuple[int, ...] = (1, 4),
    cache_modes: tuple[bool, ...] = (False, True),
) -> dict[str, int | str | bool]:
    if type(worker_counts) is not tuple or not worker_counts:
        raise ValueError("worker_counts must be a nonempty tuple")
    if type(cache_modes) is not tuple or cache_modes != (False, True):
        raise ValueError("cache_modes must be (False, True)")
    for worker in worker_counts:
        _positive_int(worker, "worker_count")
    if len(set(worker_counts)) != len(worker_counts):
        raise ValueError("worker_counts must be unique")

    materialized = tuple(rows)
    if any(type(row) is not Gate6EpisodeMetricRowV2 for row in materialized):
        raise TypeError("rows must contain only Gate6EpisodeMetricRowV2")
    if not materialized:
        return {
            "status": "not_evaluated",
            "semantic_equivalent": False,
            "group_count": 0,
            "missing_variant_count": 0,
            "duplicate_variant_count": 0,
            "semantic_mismatch_count": 0,
        }
    groups: dict[tuple[object, ...], list[Gate6EpisodeMetricRowV2]] = {}
    for row in materialized:
        key = (
            row.platform_kind,
            row.scale,
            row.ablation_case,
            row.pair_id,
            row.seed,
        )
        groups.setdefault(key, []).append(row)

    expected_variants = {
        (worker_count, cache_enabled)
        for worker_count in worker_counts
        for cache_enabled in cache_modes
    }
    missing_variant_count = 0
    duplicate_variant_count = 0
    semantic_mismatch_count = 0
    for key in sorted(groups):
        variants: dict[tuple[int, bool], Gate6EpisodeMetricRowV2] = {}
        duplicates = 0
        for row in groups[key]:
            variant = (row.worker_count, row.cache_enabled)
            if variant in variants:
                duplicates += 1
            else:
                variants[variant] = row
        duplicate_variant_count += duplicates
        missing_variant_count += len(expected_variants - set(variants))
        comparable = [variants[item] for item in sorted(set(variants) & expected_variants)]
        if comparable:
            reference = comparable[0]
            reference_semantics = (
                reference.oracle_reachable,
                reference.provider_success,
                reference.provider_complete_l2,
                reference.semantic_digest,
            )
            semantic_mismatch_count += sum(
                (
                    row.oracle_reachable,
                    row.provider_success,
                    row.provider_complete_l2,
                    row.semantic_digest,
                )
                != reference_semantics
                for row in comparable[1:]
            )
    passed = not (
        missing_variant_count or duplicate_variant_count or semantic_mismatch_count
    )
    return {
        "status": "passed" if passed else "failed",
        "semantic_equivalent": passed,
        "group_count": len(groups),
        "missing_variant_count": missing_variant_count,
        "duplicate_variant_count": duplicate_variant_count,
        "semantic_mismatch_count": semantic_mismatch_count,
    }


__all__ = [
    "GATE6_ABLATION_CASES_V2",
    "GATE6_BOOTSTRAP_RESAMPLES_V2",
    "GATE6_BOOTSTRAP_SEED_V2",
    "GATE6_HARD_TIMEOUT_MS_V2",
    "GATE6_PHASES_V2",
    "GATE6_PLATFORMS_V2",
    "GATE6_REQUIRED_INPUTS_V2",
    "Gate6EpisodeMetricRowV2",
    "Gate6FixtureAuditV2",
    "Gate6FixtureInputAuditV2",
    "Gate6FixtureInventoryV2",
    "Gate6FixtureRequirementV2",
    "aggregate_gate6_metrics_v2",
    "audit_gate6_fixture_inventory_v2",
    "audit_gate6_worker_cache_semantics_v2",
    "paired_bootstrap_coverage_ci95_v2",
]
