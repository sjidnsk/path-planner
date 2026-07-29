"""End-to-end contract tests for the offline multiscale experiment figures."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_planner_v3_multiscale.py"


def _write_artifacts(root: Path, *, include_measured_call_count: bool = True) -> None:
    summary = {
        "scenario_results": [
            {
                "scale": "TEN_METER",
                "scene": "G1_HIGH_FRONTIER",
                "platform": "WHEELED",
                "sample_count": 4,
                "correct_count": 4,
                "success_rate": 1.0,
                "p50_ns": 1_000_000,
                "p95_ns": 2_000_000,
                "maximum_ns": 3_000_000,
                "p95_target_met": True,
            }
        ],
        "scale_platform_results": [
            {
                "scale": "TEN_METER",
                "platform": "WHEELED",
                "sample_count": 4,
                "correct_count": 4,
                "success_rate": 1.0,
                "p50_ns": 1_000_000,
                "p95_ns": 2_000_000,
                "maximum_ns": 3_000_000,
                "p95_target_met": True,
            }
        ],
        "platform_results": [],
    }
    if include_measured_call_count:
        summary["measured_call_count"] = 4
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    record = {
        "scale": "TEN_METER",
        "scene": "G1_HIGH_FRONTIER",
        "platform": "WHEELED",
        "map_bounds": {"minimum_xy_m": [0.0, 0.0], "maximum_xy_m": [10.0, 10.0]},
        "start_xy_m": [1.0, 1.0],
        "goal_xy_m": [9.0, 9.0],
        "regions": [
            {
                "kind": "SYNTHETIC_TERRAIN_OBSTACLE_PROXY",
                "provenance": {"source_kind": "synthetic_terrain_obstacle_proxy/v1"},
                "polygon_xy_m": [[2.0, 7.0], [3.0, 7.0], [3.0, 8.0], [2.0, 8.0]],
            },
            {"kind": "UNKNOWN", "polygon_xy_m": [[4.0, 4.0], [9.5, 4.0], [9.5, 9.5], [4.0, 9.5]]},
        ],
        "reference_polyline_xy_m": [[1.0, 1.0], [3.0, 3.0]],
    }
    legged = {**record, "platform": "LEGGED", "reference_polyline_xy_m": [[1.0, 1.0], [3.2, 3.0]]}
    hopper = {
        **record,
        "platform": "HOPPER",
        "reference_polyline_xy_m": [[1.0, 1.0], [3.4, 3.0]],
        "ballistic_arc_xz_m": [[1.0, 0.0], [2.0, 2.0], [3.0, 0.0]],
        "landing_polygon_xy_m": [[2.5, 2.5], [3.5, 2.5], [3.5, 3.5], [2.5, 3.5]],
    }
    (root / "responses.jsonl").write_text(
        "\n".join(json.dumps(item) for item in (record, legged, hopper)) + "\n",
        encoding="utf-8",
    )


def _run_plotter(input_root: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(input_root), "--output-dir", str(output_dir)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_plotter_writes_nonempty_contract_figures(tmp_path: Path) -> None:
    """Removing either output figure must make this end-to-end contract fail."""
    _write_artifacts(tmp_path)
    output_dir = tmp_path / "figures"

    result = _run_plotter(tmp_path, output_dir)

    assert result.returncode == 0, result.stderr
    for filename in ("scenario-overview.png", "latency-results.png"):
        figure = output_dir / filename
        assert figure.is_file()
        assert figure.stat().st_size > 0
    image = mpimg.imread(output_dir / "scenario-overview.png")
    proxy_obstacle_rgb = (77 / 255, 77 / 255, 77 / 255)
    assert ((image[:, :, :3] - proxy_obstacle_rgb) ** 2).sum(axis=2).min() < 0.001


def test_plotter_rejects_summary_without_measured_call_count(tmp_path: Path) -> None:
    """Dropping the required call-count field must reject the invalid artifact."""
    _write_artifacts(tmp_path, include_measured_call_count=False)

    result = _run_plotter(tmp_path, tmp_path / "figures")

    assert result.returncode != 0
    assert "measured_call_count" in result.stderr


def test_region_polygons_accepts_runtime_proxy_kind_and_provenance() -> None:
    """The proxy kind must retain its synthetic provenance instead of becoming physical terrain."""
    region_polygons = runpy.run_path(str(SCRIPT))["_region_polygons"]
    regions = [
        {
            "kind": "SYNTHETIC_TERRAIN_OBSTACLE_PROXY",
            "provenance": {"source_kind": "synthetic_terrain_obstacle_proxy/v1"},
            "polygon_xy_m": [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
        }
    ]

    assert region_polygons(regions, "proxy_obstacles") == [
        [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)]
    ]


def test_high_frontier_panel_labels_nominal_goal_and_safe_frontier() -> None:
    """A frontier panel must distinguish an unreachable nominal goal from its safe route endpoint."""
    plotter = runpy.run_path(str(SCRIPT))
    record = {
        "scale": "TEN_METER",
        "scene": "G1_HIGH_FRONTIER",
        "platform": "WHEELED",
        "map_bounds": {"minimum_xy_m": [0.0, 0.0], "maximum_xy_m": [10.0, 10.0]},
        "start_xy_m": [1.0, 1.0],
        "goal_xy_m": [9.0, 9.0],
        "regions": [
            {
                "kind": "SYNTHETIC_TERRAIN_OBSTACLE_PROXY",
                "provenance": {"source_kind": "synthetic_terrain_obstacle_proxy/v1"},
                "polygon_xy_m": [[2.0, 7.0], [3.0, 7.0], [3.0, 8.0], [2.0, 8.0]],
            },
            {"kind": "UNKNOWN", "polygon_xy_m": [[4.0, 4.0], [9.5, 4.0], [9.5, 9.5], [4.0, 9.5]]},
        ],
        "reference_polyline_xy_m": [[1.0, 1.0], [3.0, 3.0]],
    }
    figure, axis = plt.subplots()
    try:
        plotter["_draw_scenario_panel"](axis, [record], "TEN_METER", "G1_HIGH_FRONTIER")
        labels = axis.get_legend_handles_labels()[1]
        assert "Nominal goal" in labels
        assert "Safe frontier" in labels
        assert "Proxy obstacle" in labels
    finally:
        plt.close(figure)


def test_overview_legend_includes_high_frontier_only_elements(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The figure-level legend must include labels emitted outside the first panel."""
    plotter = runpy.run_path(str(SCRIPT))
    record = {
        "scale": "TEN_METER",
        "scene": "G1_HIGH_FRONTIER",
        "platform": "WHEELED",
        "map_bounds": {"minimum_xy_m": [0.0, 0.0], "maximum_xy_m": [10.0, 10.0]},
        "start_xy_m": [1.0, 1.0],
        "goal_xy_m": [9.0, 9.0],
        "regions": [
            {
                "kind": "SYNTHETIC_TERRAIN_OBSTACLE_PROXY",
                "provenance": {"source_kind": "synthetic_terrain_obstacle_proxy/v1"},
                "polygon_xy_m": [[2.0, 7.0], [3.0, 7.0], [3.0, 8.0], [2.0, 8.0]],
            },
            {"kind": "UNKNOWN", "polygon_xy_m": [[4.0, 4.0], [9.5, 4.0], [9.5, 9.5], [4.0, 9.5]]},
        ],
        "reference_polyline_xy_m": [[1.0, 1.0], [3.0, 3.0]],
    }
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    try:
        plotter["plot_scenario_overview"]([record], tmp_path / "overview.png")
        labels = [text.get_text() for text in plt.gcf().legends[0].get_texts()]
        assert {"Nominal goal", "Safe frontier", "Unknown region", "Proxy obstacle"} <= set(labels)
    finally:
        original_close(plt.gcf())
