"""Optional Drake backend boundary for workspace IRIS prototypes."""

from .corridor_regions import build_convex_region_sequence_report
from .gcs_curvature_constrained_candidate import build_gcs_curvature_constrained_candidate_report
from .gcs_candidate import build_gcs_geometric_candidate_report
from .gcs_motion_feasibility import build_gcs_motion_feasibility_report
from .gcs_trajectory import build_gcs_trajectory_report
from .iris import build_workspace_iris_region_report
from .models import (
    ConvexRegionSequenceItem,
    ConvexRegionSequenceReport,
    GcsCurvatureConstrainedCandidateReport,
    GcsGeometricCandidateReport,
    GcsMotionFeasibilityReport,
    GcsTrajectoryReport,
    IrisRegion,
    IrisRegionReport,
)

__all__ = [
    "ConvexRegionSequenceItem",
    "ConvexRegionSequenceReport",
    "GcsCurvatureConstrainedCandidateReport",
    "GcsGeometricCandidateReport",
    "GcsMotionFeasibilityReport",
    "GcsTrajectoryReport",
    "IrisRegion",
    "IrisRegionReport",
    "build_convex_region_sequence_report",
    "build_gcs_curvature_constrained_candidate_report",
    "build_gcs_geometric_candidate_report",
    "build_gcs_motion_feasibility_report",
    "build_gcs_trajectory_report",
    "build_workspace_iris_region_report",
]
