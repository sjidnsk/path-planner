from __future__ import annotations

from dataclasses import replace

import pytest

from path_planner.v2 import benchmark_fixtures as gate6


EXPECTED_INPUTS = (
    (
        "independent_primitive_labels",
        "provide_independent_10000_primitive_labels_per_platform",
        10_000,
    ),
    (
        "independent_small_map_optima",
        "provide_independent_small_map_optima",
        1,
    ),
    (
        "standard_schedules",
        "provide_standard_100_episodes_per_platform",
        100,
    ),
    (
        "kilometer_schedules",
        "provide_kilometer_30_episodes_per_platform",
        30,
    ),
    ("ppo_targets", "provide_ppo_target_fixtures", 1),
)


def _row(**overrides: object) -> gate6.Gate6EpisodeMetricRowV2:
    payload: dict[str, object] = {
        "episode_id": "episode-001",
        "pair_id": "pair-001",
        "platform_kind": "wheel",
        "scale": "standard",
        "ablation_case": "v2_full",
        "seed": 7,
        "worker_count": 1,
        "cache_enabled": False,
        "oracle_reachable": True,
        "provider_success": True,
        "provider_complete_l2": True,
        "primitive_true_positive_count": 9,
        "primitive_false_negative_count": 1,
        "primitive_false_positive_count": 0,
        "resource_cost": 10.0,
        "coverage_efficiency": 4.0,
        "expanded_states": 20,
        "rejected_l0": 1,
        "rejected_l1": 2,
        "rejected_l2": 3,
        "cache_hits": 1,
        "cache_lookups": 2,
        "runtime_ms": 30.0,
        "timed_out": False,
        "semantic_digest": "a" * 64,
    }
    payload.update(overrides)
    return gate6.Gate6EpisodeMetricRowV2(**payload)


def test_gate6_freezes_inputs_phases_and_complete_ablation_matrix() -> None:
    assert gate6.GATE6_PLATFORMS_V2 == ("wheel", "legged", "hopper")
    assert tuple(
        (item.input_kind, item.blocker, item.minimum_rows_per_platform)
        for item in gate6.GATE6_REQUIRED_INPUTS_V2
    ) == EXPECTED_INPUTS
    assert all(item.requires_independent_source for item in gate6.GATE6_REQUIRED_INPUTS_V2)
    assert gate6.GATE6_PHASES_V2 == (
        "primitive_audit",
        "exact_quality",
        "standard",
        "kilometer",
        "ablation",
        "determinism_worker_cache",
        "aggregate",
    )
    assert gate6.GATE6_ABLATION_CASES_V2 == (
        "v1_astar",
        "wheel_hybrid_astar_opt_in",
        "v2_fine_only",
        "v2_plus_multi_heuristic",
        "v2_plus_hierarchy",
        "v2_plus_lazy_validation",
        "v2_plus_cache",
        "v2_full",
    )


def test_gate6_inventory_audit_has_stable_blockers_and_rejects_self_labels() -> None:
    missing = {
        item.input_kind: None for item in reversed(gate6.GATE6_REQUIRED_INPUTS_V2)
    }
    audit = gate6.audit_gate6_fixture_inventory_v2(missing)
    assert audit.blocking_reasons == tuple(item[1] for item in EXPECTED_INPUTS)
    assert audit.formal_evidence_eligible is False
    assert audit.formal_row_count == 0
    assert tuple(item.status for item in audit.inputs) == ("missing",) * 5

    inventories = {}
    for input_kind, _, minimum in EXPECTED_INPUTS:
        inventories[input_kind] = gate6.Gate6FixtureInventoryV2(
            input_kind=input_kind,
            source_id=f"independent-{input_kind}/v1",
            source_independent=input_kind != "independent_primitive_labels",
            rows_per_platform=tuple(
                (platform, minimum) for platform in gate6.GATE6_PLATFORMS_V2
            ),
        )
    self_labelled = gate6.audit_gate6_fixture_inventory_v2(inventories)
    assert self_labelled.blocking_reasons == (EXPECTED_INPUTS[0][1],)
    assert self_labelled.inputs[0].status == "non_independent"
    assert self_labelled.formal_evidence_eligible is False


def test_gate6_metric_aggregate_covers_all_required_statistics() -> None:
    rows = [
        _row(episode_id="episode-001", runtime_ms=10.0),
        _row(
            episode_id="episode-002",
            pair_id="pair-002",
            provider_complete_l2=False,
            primitive_true_positive_count=8,
            primitive_false_negative_count=2,
            primitive_false_positive_count=1,
            resource_cost=12.0,
            coverage_efficiency=2.0,
            expanded_states=30,
            rejected_l0=2,
            rejected_l1=3,
            rejected_l2=4,
            cache_hits=2,
            cache_lookups=4,
            runtime_ms=30.0,
        ),
        _row(
            episode_id="episode-003",
            pair_id="pair-003",
            oracle_reachable=False,
            provider_success=False,
            provider_complete_l2=False,
            primitive_true_positive_count=0,
            primitive_false_negative_count=0,
            resource_cost=None,
            coverage_efficiency=0.0,
            expanded_states=10,
            rejected_l0=0,
            rejected_l1=0,
            rejected_l2=1,
            cache_hits=0,
            cache_lookups=0,
            runtime_ms=2000.001,
            timed_out=True,
        ),
    ]

    summary = gate6.aggregate_gate6_metrics_v2(reversed(rows))

    assert summary["metrics_status"] == "evaluated"
    assert summary["row_count"] == 3
    assert summary["unsafe_success_count"] == 1
    assert summary["primitive_false_positive_count"] == 1
    assert summary["primitive_recall"] == pytest.approx(17 / 20)
    assert summary["reachable_success_ratio"] == 1.0
    assert summary["mean_resource_cost"] == 11.0
    assert summary["mean_coverage_efficiency"] == 2.0
    assert summary["expanded_states"] == 60
    assert (summary["rejected_l0"], summary["rejected_l1"], summary["rejected_l2"]) == (3, 5, 8)
    assert summary["cache_hit_ratio"] == 0.5
    assert (summary["runtime_p50_ms"], summary["runtime_p95_ms"], summary["runtime_p99_ms"]) == (30.0, 2000.001, 2000.001)
    assert summary["timeout_count"] == 1
    assert summary["hard_timeout_violation_count"] == 1


def test_gate6_failed_episode_requires_coverage_but_forbids_resource_cost() -> None:
    failed = _row(
        provider_success=False,
        provider_complete_l2=False,
        resource_cost=None,
        coverage_efficiency=0.0,
    )
    assert failed.coverage_efficiency == 0.0

    with pytest.raises(ValueError, match="coverage_efficiency"):
        replace(failed, coverage_efficiency=None)
    with pytest.raises(ValueError, match="resource_cost"):
        replace(failed, resource_cost=1.0)


def test_gate6_empty_metric_and_semantics_audits_are_not_evaluated() -> None:
    summary = gate6.aggregate_gate6_metrics_v2(())
    assert summary["metrics_status"] == "not_evaluated"
    assert summary["row_count"] == 0
    assert summary["safety_passed"] is None

    audit = gate6.audit_gate6_worker_cache_semantics_v2(())
    assert audit["status"] == "not_evaluated"
    assert audit["semantic_equivalent"] is False
    assert audit["group_count"] == 0


def test_gate6_paired_bootstrap_and_worker_cache_audit_are_deterministic() -> None:
    rows = []
    for case_id, coverage, digest in (
        ("v2_fine_only", 2.0, "a" * 64),
        ("v2_full", 4.0, "b" * 64),
    ):
        for worker_count in (1, 4):
            for cache_enabled in (False, True):
                rows.append(
                    _row(
                        episode_id=f"{case_id}-{worker_count}-{int(cache_enabled)}",
                        ablation_case=case_id,
                        worker_count=worker_count,
                        cache_enabled=cache_enabled,
                        coverage_efficiency=coverage,
                        semantic_digest=digest,
                    )
                )

    forward = gate6.paired_bootstrap_coverage_ci95_v2(
        rows,
        baseline_case="v2_fine_only",
        candidate_case="v2_full",
        seed=20260716,
        resamples=128,
    )
    reverse = gate6.paired_bootstrap_coverage_ci95_v2(
        reversed(rows),
        baseline_case="v2_fine_only",
        candidate_case="v2_full",
        seed=20260716,
        resamples=128,
    )
    assert forward == reverse
    assert forward["pair_count"] == 1
    assert forward["mean_delta"] == 2.0
    assert (forward["ci95_lower"], forward["ci95_upper"]) == (2.0, 2.0)
    assert gate6.audit_gate6_worker_cache_semantics_v2(rows)["status"] == "passed"

    drifted = list(rows)
    drifted[-1] = replace(drifted[-1], semantic_digest="c" * 64)
    audit = gate6.audit_gate6_worker_cache_semantics_v2(drifted)
    assert audit["status"] == "failed"
    assert audit["semantic_mismatch_count"] == 1
