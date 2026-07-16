from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
from importlib import import_module

import pytest


def _v2():
    return import_module("path_planner.v2")


def _primitive_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "row_id": "primitive-001",
        "seed": 7,
        "expected_safe": True,
        "expected_label_source": "independent-wheel-oracle/v1",
        "expected_label_independent": True,
        "provider_safe": True,
        "provider_complete_l2": True,
        "runtime_ms": 10.0,
        "timed_out": False,
        "reason_code": "safe",
    }
    payload.update(overrides)
    return payload


def _exact_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "row_id": "exact-001",
        "seed": 7,
        "optimum_resource_cost": 10.0,
        "optimum_source": "independent-exact-solver/v1",
        "optimum_independent": True,
        "provider_success": True,
        "provider_resource_cost": 10.5,
        "provider_complete_l2": True,
        "runtime_ms": 20.0,
        "timed_out": False,
        "reason_code": "goal_reached",
    }
    payload.update(overrides)
    return payload


def _standard_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "episode_id": "standard-001",
        "seed": 7,
        "schedule_source": "independent-standard-wheel/v1",
        "schedule_independent": True,
        "oracle_reachable": True,
        "provider_success": True,
        "provider_complete_l2": True,
        "runtime_ms": 30.0,
        "timed_out": False,
        "reason_code": "goal_reached",
    }
    payload.update(overrides)
    return payload


def test_public_benchmark_rows_are_exact_frozen_slotted_contracts():
    v2 = _v2()

    assert tuple(field.name for field in fields(v2.PrimitiveAuditRowV2)) == tuple(
        _primitive_payload()
    )
    assert tuple(field.name for field in fields(v2.ExactMapQualityRowV2)) == tuple(
        _exact_payload()
    )
    assert tuple(field.name for field in fields(v2.StandardEpisodeRowV2)) == tuple(
        _standard_payload()
    )

    row = v2.PrimitiveAuditRowV2.from_dict(_primitive_payload())
    assert not hasattr(row, "__dict__")
    with pytest.raises(FrozenInstanceError):
        row.provider_safe = False


def test_row_parsing_preserves_independent_labels_and_provider_results_separately():
    v2 = _v2()

    false_positive = v2.PrimitiveAuditRowV2.from_dict(
        _primitive_payload(expected_safe=False, provider_safe=True)
    )
    false_negative = v2.PrimitiveAuditRowV2.from_dict(
        _primitive_payload(
            row_id="primitive-002",
            expected_safe=True,
            provider_safe=False,
            provider_complete_l2=False,
            reason_code="provider_rejected",
        )
    )
    quality = v2.ExactMapQualityRowV2.from_dict(
        _exact_payload(optimum_resource_cost=4.0, provider_resource_cost=5.0)
    )

    assert (false_positive.expected_safe, false_positive.provider_safe) == (False, True)
    assert (false_negative.expected_safe, false_negative.provider_safe) == (True, False)
    assert quality.optimum_resource_cost == 4.0
    assert quality.provider_resource_cost == 5.0


@pytest.mark.parametrize("provider_resource_cost", [0.0, 5.0])
def test_exact_row_rejects_success_cost_below_the_independent_optimum(
    provider_resource_cost,
):
    v2 = _v2()

    with pytest.raises(ValueError, match="provider_resource_cost.*optimum_resource_cost"):
        v2.ExactMapQualityRowV2.from_dict(
            _exact_payload(
                optimum_resource_cost=10.0,
                optimum_independent=True,
                provider_resource_cost=provider_resource_cost,
            )
        )


@pytest.mark.parametrize("provider_resource_cost", [10.0, 11.0])
def test_exact_row_accepts_optimum_and_one_point_one_cost_ratio(provider_resource_cost):
    v2 = _v2()
    row = v2.ExactMapQualityRowV2.from_dict(
        _exact_payload(
            optimum_resource_cost=10.0,
            provider_resource_cost=provider_resource_cost,
        )
    )

    summary = v2.aggregate_exact_map_quality_v2([row])

    assert summary.max_resource_cost_ratio == provider_resource_cost / 10.0


@pytest.mark.parametrize(
    ("factory_name", "payload_factory"),
    [
        ("PrimitiveAuditRowV2", _primitive_payload),
        ("ExactMapQualityRowV2", _exact_payload),
        ("StandardEpisodeRowV2", _standard_payload),
    ],
)
def test_row_parsing_rejects_missing_and_extra_keys(factory_name, payload_factory):
    v2 = _v2()
    row_type = getattr(v2, factory_name)
    missing = payload_factory()
    missing.pop(next(iter(missing)))
    extra = payload_factory(extra_field="not-allowed")

    with pytest.raises(ValueError, match="exact keys"):
        row_type.from_dict(missing)
    with pytest.raises(ValueError, match="exact keys"):
        row_type.from_dict(extra)


@pytest.mark.parametrize(
    ("factory_name", "payload"),
    [
        ("PrimitiveAuditRowV2", _primitive_payload(expected_safe=1)),
        ("PrimitiveAuditRowV2", _primitive_payload(seed=True)),
        ("PrimitiveAuditRowV2", _primitive_payload(runtime_ms=True)),
        ("PrimitiveAuditRowV2", _primitive_payload(runtime_ms=-0.1)),
        ("PrimitiveAuditRowV2", _primitive_payload(runtime_ms=float("nan"))),
        ("ExactMapQualityRowV2", _exact_payload(optimum_independent=1)),
        ("ExactMapQualityRowV2", _exact_payload(optimum_resource_cost=0.0)),
        ("ExactMapQualityRowV2", _exact_payload(provider_resource_cost=float("inf"))),
        ("StandardEpisodeRowV2", _standard_payload(schedule_independent=1)),
        ("StandardEpisodeRowV2", _standard_payload(runtime_ms=float("-inf"))),
    ],
)
def test_rows_reject_bool_number_confusion_and_invalid_numeric_values(factory_name, payload):
    v2 = _v2()

    with pytest.raises((TypeError, ValueError)):
        getattr(v2, factory_name).from_dict(payload)


@pytest.mark.parametrize(
    ("factory_name", "payload"),
    [
        ("PrimitiveAuditRowV2", _primitive_payload(runtime_ms=10**400)),
        ("ExactMapQualityRowV2", _exact_payload(optimum_resource_cost=10**400)),
        ("ExactMapQualityRowV2", _exact_payload(provider_resource_cost=10**400)),
        ("StandardEpisodeRowV2", _standard_payload(runtime_ms=10**400)),
    ],
)
def test_rows_normalize_float_overflow_to_a_stable_finite_value_error(factory_name, payload):
    v2 = _v2()

    with pytest.raises(ValueError, match="finite"):
        getattr(v2, factory_name).from_dict(payload)


@pytest.mark.parametrize(
    ("factory_name", "payload"),
    [
        ("PrimitiveAuditRowV2", _primitive_payload(row_id="contains space")),
        ("PrimitiveAuditRowV2", _primitive_payload(expected_label_source="\nsource")),
        ("ExactMapQualityRowV2", _exact_payload(optimum_source="bad source")),
        ("StandardEpisodeRowV2", _standard_payload(episode_id="")),
        ("StandardEpisodeRowV2", _standard_payload(reason_code="bad reason")),
    ],
)
def test_rows_reject_malformed_stable_identifiers_and_sources(factory_name, payload):
    v2 = _v2()

    with pytest.raises(ValueError, match="stable"):
        getattr(v2, factory_name).from_dict(payload)


@pytest.mark.parametrize(
    ("factory_name", "payload", "message"),
    [
        (
            "PrimitiveAuditRowV2",
            _primitive_payload(provider_safe=False, provider_complete_l2=True),
            "complete L2",
        ),
        (
            "PrimitiveAuditRowV2",
            _primitive_payload(timed_out=True),
            "timed out",
        ),
        (
            "ExactMapQualityRowV2",
            _exact_payload(provider_success=True, provider_resource_cost=None),
            "resource cost",
        ),
        (
            "ExactMapQualityRowV2",
            _exact_payload(
                provider_success=False,
                provider_resource_cost=3.0,
                provider_complete_l2=False,
            ),
            "resource cost",
        ),
        (
            "ExactMapQualityRowV2",
            _exact_payload(
                provider_success=False,
                provider_resource_cost=None,
                provider_complete_l2=True,
            ),
            "complete L2",
        ),
        (
            "StandardEpisodeRowV2",
            _standard_payload(provider_success=False, provider_complete_l2=True),
            "complete L2",
        ),
        (
            "StandardEpisodeRowV2",
            _standard_payload(timed_out=True),
            "timed out",
        ),
    ],
)
def test_rows_reject_inconsistent_provider_outcomes(factory_name, payload, message):
    v2 = _v2()

    with pytest.raises(ValueError, match=message):
        getattr(v2, factory_name).from_dict(payload)


def test_primitive_aggregation_excludes_self_labeled_rows_from_all_formal_metrics():
    v2 = _v2()
    formal_true_positive = v2.PrimitiveAuditRowV2.from_dict(_primitive_payload())
    formal_false_negative = v2.PrimitiveAuditRowV2.from_dict(
        _primitive_payload(
            row_id="primitive-002",
            expected_safe=True,
            provider_safe=False,
            provider_complete_l2=False,
            runtime_ms=20.0,
            reason_code="provider_rejected",
        )
    )
    excluded_false_positive_timeout = v2.PrimitiveAuditRowV2.from_dict(
        _primitive_payload(
            row_id="primitive-003",
            expected_safe=False,
            expected_label_source="provider-self-label/v1",
            expected_label_independent=False,
            provider_safe=False,
            provider_complete_l2=False,
            runtime_ms=9999.0,
            timed_out=True,
            reason_code="timeout",
        )
    )

    summary = v2.aggregate_primitive_audit_v2(
        [excluded_false_positive_timeout, formal_false_negative, formal_true_positive]
    )

    assert tuple(row.row_id for row in summary.rows) == (
        "primitive-001",
        "primitive-002",
        "primitive-003",
    )
    assert summary.total_rows == 3
    assert summary.formal_row_count == 2
    assert summary.excluded_row_count == 1
    assert summary.false_positive_count == 0
    assert summary.true_positive_count == 1
    assert summary.false_negative_count == 1
    assert summary.primitive_recall == 0.5
    assert summary.provider_success_count == 1
    assert summary.complete_l2_ratio == 1.0
    assert (summary.runtime_p50_ms, summary.runtime_p95_ms, summary.runtime_p99_ms) == (
        10.0,
        20.0,
        20.0,
    )
    assert summary.timeout_count == 0
    assert summary.hard_timeout_violation_count == 0
    assert summary.reason_histogram == (("provider_rejected", 1), ("safe", 1))


def test_primitive_false_positive_and_nearest_rank_percentiles_are_exact():
    v2 = _v2()
    runtimes = (50.0, 10.0, 30.0, 20.0, 40.0)
    rows = [
        v2.PrimitiveAuditRowV2.from_dict(
            _primitive_payload(
                row_id=f"primitive-{index:03d}",
                seed=5 - index,
                expected_safe=index != 1,
                provider_safe=index != 2,
                provider_complete_l2=index != 2,
                runtime_ms=runtime,
                reason_code="safe" if index != 2 else "provider_rejected",
            )
        )
        for index, runtime in enumerate(runtimes, start=1)
    ]

    summary = v2.aggregate_primitive_audit_v2(rows)

    assert tuple((row.seed, row.row_id) for row in summary.rows) == tuple(
        sorted((row.seed, row.row_id) for row in rows)
    )
    assert summary.false_positive_count == 1
    assert summary.primitive_recall == pytest.approx(0.75)
    assert summary.runtime_p50_ms == 30.0
    assert summary.runtime_p95_ms == 50.0
    assert summary.runtime_p99_ms == 50.0


def test_exact_aggregation_keeps_independent_optimum_ratios_and_excludes_self_labels():
    v2 = _v2()
    rows = [
        v2.ExactMapQualityRowV2.from_dict(
            _exact_payload(row_id="exact-002", seed=2, provider_resource_cost=11.0)
        ),
        v2.ExactMapQualityRowV2.from_dict(
            _exact_payload(
                row_id="exact-001",
                seed=1,
                optimum_resource_cost=4.0,
                provider_resource_cost=5.0,
                runtime_ms=10.0,
            )
        ),
        v2.ExactMapQualityRowV2.from_dict(
            _exact_payload(
                row_id="exact-003",
                seed=3,
                optimum_source="provider-self-label/v1",
                optimum_independent=False,
                optimum_resource_cost=1.0,
                provider_resource_cost=100.0,
                runtime_ms=9999.0,
            )
        ),
    ]

    summary = v2.aggregate_exact_map_quality_v2(rows)

    assert tuple(row.row_id for row in summary.rows) == ("exact-001", "exact-002", "exact-003")
    assert summary.total_rows == 3
    assert summary.formal_row_count == 2
    assert summary.excluded_row_count == 1
    assert summary.provider_success_count == 2
    assert summary.provider_success_ratio == 1.0
    assert summary.complete_l2_ratio == 1.0
    assert summary.resource_cost_ratios == (("exact-001", 1.25), ("exact-002", 1.1))
    assert summary.max_resource_cost_ratio == 1.25
    assert summary.runtime_p50_ms == 10.0
    assert summary.runtime_p95_ms == 20.0
    assert summary.reason_histogram == (("goal_reached", 2),)


def test_exact_aggregation_rejects_a_nonfinite_derived_cost_ratio():
    v2 = _v2()
    row = v2.ExactMapQualityRowV2.from_dict(
        _exact_payload(
            optimum_resource_cost=1e-308,
            provider_resource_cost=1e308,
        )
    )

    with pytest.raises(ValueError, match="resource cost ratio.*finite"):
        v2.aggregate_exact_map_quality_v2([row])


def test_standard_aggregation_uses_only_independent_reachable_denominator():
    v2 = _v2()
    rows = [
        v2.StandardEpisodeRowV2.from_dict(_standard_payload()),
        v2.StandardEpisodeRowV2.from_dict(
            _standard_payload(
                episode_id="standard-002",
                seed=8,
                provider_success=False,
                provider_complete_l2=False,
                runtime_ms=40.0,
                reason_code="no_complete_route",
            )
        ),
        v2.StandardEpisodeRowV2.from_dict(
            _standard_payload(
                episode_id="standard-003",
                seed=9,
                oracle_reachable=False,
                provider_success=False,
                provider_complete_l2=False,
                runtime_ms=50.0,
                reason_code="oracle_unreachable",
            )
        ),
        v2.StandardEpisodeRowV2.from_dict(
            _standard_payload(
                episode_id="standard-004",
                seed=10,
                schedule_source="provider-self-schedule/v1",
                schedule_independent=False,
                runtime_ms=9999.0,
                timed_out=True,
                provider_success=False,
                provider_complete_l2=False,
                reason_code="timeout",
            )
        ),
    ]

    summary = v2.aggregate_standard_episodes_v2(rows)

    assert summary.total_episodes == 4
    assert summary.formal_episode_count == 3
    assert summary.excluded_episode_count == 1
    assert summary.formal_oracle_reachable_count == 2
    assert summary.reachable_provider_success_count == 1
    assert summary.reachable_query_success_ratio == 0.5
    assert summary.complete_l2_ratio == 1.0
    assert (summary.runtime_p50_ms, summary.runtime_p95_ms, summary.runtime_p99_ms) == (
        40.0,
        50.0,
        50.0,
    )
    assert summary.timeout_count == 0
    assert summary.hard_timeout_violation_count == 0
    assert summary.reason_histogram == (
        ("goal_reached", 1),
        ("no_complete_route", 1),
        ("oracle_unreachable", 1),
    )


@pytest.mark.parametrize(
    ("aggregate_name", "rows", "duplicate_id"),
    [
        (
            "aggregate_primitive_audit_v2",
            [
                _primitive_payload(row_id="duplicate", seed=1),
                _primitive_payload(row_id="duplicate", seed=2),
            ],
            "duplicate",
        ),
        (
            "aggregate_exact_map_quality_v2",
            [
                _exact_payload(row_id="duplicate", seed=1),
                _exact_payload(row_id="duplicate", seed=2),
            ],
            "duplicate",
        ),
        (
            "aggregate_standard_episodes_v2",
            [
                _standard_payload(episode_id="duplicate", seed=1),
                _standard_payload(episode_id="duplicate", seed=2),
            ],
            "duplicate",
        ),
    ],
)
def test_aggregations_reject_duplicate_stable_ids(aggregate_name, rows, duplicate_id):
    v2 = _v2()
    type_by_aggregate = {
        "aggregate_primitive_audit_v2": v2.PrimitiveAuditRowV2,
        "aggregate_exact_map_quality_v2": v2.ExactMapQualityRowV2,
        "aggregate_standard_episodes_v2": v2.StandardEpisodeRowV2,
    }
    typed_rows = [type_by_aggregate[aggregate_name].from_dict(row) for row in rows]

    with pytest.raises(ValueError, match=duplicate_id):
        getattr(v2, aggregate_name)(typed_rows)


def test_empty_formal_denominators_are_explicitly_unavailable():
    v2 = _v2()
    primitive = v2.aggregate_primitive_audit_v2(
        [
            v2.PrimitiveAuditRowV2.from_dict(
                _primitive_payload(expected_label_independent=False)
            )
        ]
    )
    exact = v2.aggregate_exact_map_quality_v2(
        [v2.ExactMapQualityRowV2.from_dict(_exact_payload(optimum_independent=False))]
    )
    standard = v2.aggregate_standard_episodes_v2(
        [v2.StandardEpisodeRowV2.from_dict(_standard_payload(schedule_independent=False))]
    )

    assert primitive.primitive_recall is None
    assert primitive.complete_l2_ratio is None
    assert primitive.runtime_p50_ms is None
    assert exact.provider_success_ratio is None
    assert exact.max_resource_cost_ratio is None
    assert exact.runtime_p95_ms is None
    assert standard.reachable_query_success_ratio is None
    assert standard.complete_l2_ratio is None
    assert standard.runtime_p99_ms is None


def test_hard_timeout_violations_use_strictly_greater_than_two_seconds():
    v2 = _v2()
    primitive = v2.aggregate_primitive_audit_v2(
        [
            v2.PrimitiveAuditRowV2.from_dict(
                _primitive_payload(row_id="at-limit", runtime_ms=2000.0)
            ),
            v2.PrimitiveAuditRowV2.from_dict(
                _primitive_payload(row_id="over-limit", seed=8, runtime_ms=2000.001)
            ),
        ]
    )
    exact = v2.aggregate_exact_map_quality_v2(
        [v2.ExactMapQualityRowV2.from_dict(_exact_payload(runtime_ms=2000.001))]
    )
    standard = v2.aggregate_standard_episodes_v2(
        [v2.StandardEpisodeRowV2.from_dict(_standard_payload(runtime_ms=2000.001))]
    )

    assert primitive.hard_timeout_violation_count == 1
    assert exact.hard_timeout_violation_count == 1
    assert standard.hard_timeout_violation_count == 1


def test_stable_rows_and_summary_dicts_are_deterministic_digest_inputs():
    v2 = _v2()
    rows = [
        v2.PrimitiveAuditRowV2.from_dict(
            _primitive_payload(row_id="primitive-b", seed=2, reason_code="z-reason")
        ),
        v2.PrimitiveAuditRowV2.from_dict(
            _primitive_payload(row_id="primitive-a", seed=1, reason_code="a-reason")
        ),
    ]

    forward = v2.aggregate_primitive_audit_v2(rows)
    reverse = v2.aggregate_primitive_audit_v2(reversed(rows))
    forward_bytes = v2.canonical_json_bytes(
        {
            "rows": [row.to_dict() for row in forward.rows],
            "summary": forward.to_dict(),
        }
    )
    reverse_bytes = v2.canonical_json_bytes(
        {
            "rows": [row.to_dict() for row in reverse.rows],
            "summary": reverse.to_dict(),
        }
    )

    assert forward.to_dict()["reason_histogram"] == {"a-reason": 1, "z-reason": 1}
    assert forward_bytes == reverse_bytes
    assert sha256(forward_bytes).hexdigest() == sha256(reverse_bytes).hexdigest()
