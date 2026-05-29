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
