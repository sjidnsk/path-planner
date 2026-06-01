"""Drake-free region graph models for future IRIS/GCS backends."""

from .grid import (
    build_blocked_cell_obstacles,
    build_grid_box_regions,
    build_region_edges,
    build_region_graph_report,
)
from .models import (
    ConvexRegion,
    ObstaclePrimitive,
    RegionEdge,
    RegionGraph,
    RegionGraphReport,
)

__all__ = [
    "ConvexRegion",
    "ObstaclePrimitive",
    "RegionEdge",
    "RegionGraph",
    "RegionGraphReport",
    "build_blocked_cell_obstacles",
    "build_grid_box_regions",
    "build_region_edges",
    "build_region_graph_report",
]
