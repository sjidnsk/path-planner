"""End-to-end contract tests for the offline multiscale experiment figures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_planner_v3_multiscale.py"


def _write_artifacts(root: Path, *, include_measured_call_count: bool = True) -> None:
    summary = {
        "scenario_results": [
            {
                "scale": "TEN_METER",
                "scene": "OPEN",
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
        "scene": "OPEN",
        "platform": "HOPPER",
        "map_bounds": {"minimum_xy_m": [0.0, 0.0], "maximum_xy_m": [10.0, 10.0]},
        "start_xy_m": [1.0, 1.0],
        "goal_xy_m": [9.0, 9.0],
        "regions": [
            {"kind": "known_terrain", "polygon_xy_m": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]},
            {"kind": "unknown_region", "polygon_xy_m": [[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]},
            {"kind": "obstacle", "polygon_xy_m": [[2.0, 7.0], [3.0, 7.0], [3.0, 8.0], [2.0, 8.0]]},
        ],
        "reference_polyline_xy_m": [[1.0, 1.0], [5.0, 3.0], [9.0, 9.0]],
        "ballistic_arc_xz_m": [[1.0, 0.0], [5.0, 3.0], [9.0, 0.0]],
        "landing_polygon_xy_m": [[8.5, 8.5], [9.5, 8.5], [9.5, 9.5], [8.5, 9.5]],
    }
    (root / "responses.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


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
    known_terrain_rgb = (217 / 255, 199 / 255, 162 / 255)
    assert ((image[:, :, :3] - known_terrain_rgb) ** 2).sum(axis=2).min() < 0.001


def test_plotter_rejects_summary_without_measured_call_count(tmp_path: Path) -> None:
    """Dropping the required call-count field must reject the invalid artifact."""
    _write_artifacts(tmp_path, include_measured_call_count=False)

    result = _run_plotter(tmp_path, tmp_path / "figures")

    assert result.returncode != 0
    assert "measured_call_count" in result.stderr
