from __future__ import annotations

from typing import Any

from path_planner.tracking import TrackingSimulationResult


TRACKING_COMPARISON_METRICS = (
    "path_length_m",
    "simulated_length_m",
    "max_cross_track_error_m",
    "min_clearance_m",
    "safety_violation_count",
    "curvature_violation_count",
    "high_cost_exposure",
)


def build_tracking_metric_comparison(
    baseline: TrackingSimulationResult | None,
    optimized: TrackingSimulationResult | None,
) -> dict[str, Any]:
    if baseline is None or optimized is None:
        return {}
    baseline_metrics = baseline.metrics.to_dict()
    optimized_metrics = optimized.metrics.to_dict()
    comparison: dict[str, Any] = {}
    for metric in TRACKING_COMPARISON_METRICS:
        baseline_value = baseline_metrics[metric]
        optimized_value = optimized_metrics[metric]
        comparison[metric] = {
            "baseline": baseline_value,
            "optimized": optimized_value,
            "delta": _delta(optimized_value, baseline_value),
        }
    return comparison


def merge_tracking_comparison(
    trajectory_optimization_payload: dict[str, Any],
    baseline: TrackingSimulationResult | None,
    optimized: TrackingSimulationResult | None,
) -> dict[str, Any]:
    comparison = build_tracking_metric_comparison(baseline, optimized)
    if comparison:
        trajectory_optimization_payload.setdefault("metrics", {}).setdefault("baseline_vs_optimized", {}).update(comparison)
    return trajectory_optimization_payload


def _delta(optimized: Any, baseline: Any) -> float | int | None:
    if optimized is None or baseline is None:
        return None
    return optimized - baseline
