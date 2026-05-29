from __future__ import annotations

from typing import Mapping

import numpy as np

from path_planner.core import CostGrid, GridSpec


class DevPlatformAdapter:
    def from_arrays(
        self,
        *,
        cost: np.ndarray,
        passable_mask: np.ndarray,
        resolution: float,
        origin: tuple[float, float] = (0.0, 0.0),
        frame_id: str = "map",
        layers: Mapping[str, np.ndarray] | None = None,
    ) -> CostGrid:
        cost_array = np.asarray(cost, dtype=float)
        if cost_array.ndim != 2:
            raise ValueError("cost must be a 2D array")
        spec = GridSpec(
            width=cost_array.shape[1],
            height=cost_array.shape[0],
            resolution=resolution,
            origin=origin,
            frame_id=frame_id,
        )
        layer_names = sorted((layers or {}).keys())
        return CostGrid(
            spec=spec,
            cost=cost_array,
            passable_mask=np.asarray(passable_mask, dtype=bool),
            metadata={
                "adapter": "dev-platform-constraints",
                "layers": layer_names,
            },
        )
