import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.diagnostics import render_diagnostics
from path_planner.search import AStarPlanner


def test_render_diagnostics_writes_png_and_html(tmp_path):
    spec = GridSpec(width=3, height=3, resolution=1.0)
    grid = CostGrid(spec=spec, cost=np.ones((3, 3)), passable_mask=np.ones((3, 3), dtype=bool))
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 2)))

    png_path = tmp_path / "diagnostics.png"
    html_path = tmp_path / "diagnostics.html"
    render_diagnostics(grid, result, png_path=png_path, html_path=html_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    assert "Cost + Path" in html
    assert "trajectory_kind" in html
    assert "geometric_path" in html


def test_cli_demo_writes_json_png_and_html(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            "examples/demo_map.json",
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert (output_dir / "diagnostics.png").exists()
    assert (output_dir / "diagnostics.html").exists()
    assert "reachable" in completed.stdout
