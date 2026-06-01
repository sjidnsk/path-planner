"""Optional Drake backend boundary for workspace IRIS prototypes."""

from .iris import build_workspace_iris_region_report
from .models import IrisRegion, IrisRegionReport

__all__ = [
    "IrisRegion",
    "IrisRegionReport",
    "build_workspace_iris_region_report",
]
