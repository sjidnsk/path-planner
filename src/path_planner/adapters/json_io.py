from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest, PlanResult
from path_planner.postprocess import PostprocessResult
from path_planner.tracking import TrackingSimulationResult

REQUEST_SCHEMA_VERSION = "path-planner-request/v1"


def load_plan_input(path: str | Path) -> tuple[CostGrid, PlanRequest]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {REQUEST_SCHEMA_VERSION}")
    grid_raw = raw["grid"]
    origin = grid_raw.get("origin", [0.0, 0.0])
    spec = GridSpec(
        width=int(grid_raw["width"]),
        height=int(grid_raw["height"]),
        resolution=float(grid_raw["resolution"]),
        origin=(float(origin[0]), float(origin[1])),
        frame_id=str(grid_raw.get("frame_id", "map")),
    )
    grid = CostGrid(
        spec=spec,
        cost=np.asarray(raw["cost"], dtype=float),
        passable_mask=np.asarray(raw["passable_mask"], dtype=bool),
        metadata={"adapter": "json", "source": str(path)},
    )
    request = PlanRequest(
        start=Cell(int(raw["start"][0]), int(raw["start"][1])),
        goal=Cell(int(raw["goal"][0]), int(raw["goal"][1])),
        max_iterations=int(raw.get("max_iterations", 100_000)),
    )
    return grid, request


def route_result_to_json_dict(
    result: PlanResult,
    spec: GridSpec,
    *,
    postprocess: PostprocessResult | None = None,
    tracking_simulation: TrackingSimulationResult | None = None,
) -> dict[str, Any]:
    payload = result.to_route_dict(spec)
    if postprocess is not None:
        payload["postprocess"] = postprocess.to_dict()
    if tracking_simulation is not None:
        payload["tracking_simulation_report"] = tracking_simulation.to_dict()
    return payload
