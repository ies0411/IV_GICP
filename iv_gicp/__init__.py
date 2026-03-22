"""IV-GICP: Intensity-Augmented GICP with C++ registration and voxel map (Python bindings)."""

from .iv_gicp import IVGICP
from .pipeline import IVGICPPipeline, OdometryResult, Trajectory
from .odometry import IVGICPOdometry
from .degeneracy_analysis import (
    DegeneracyMonitor,
    DegeneracyReport,
    analyze_degeneracy,
    check_photometric_rescue,
    compute_geometric_fim,
    compute_photometric_fim,
)

__all__ = [
    "IVGICP",
    "IVGICPOdometry",
    "IVGICPPipeline",
    "OdometryResult",
    "Trajectory",
    "DegeneracyMonitor",
    "DegeneracyReport",
    "analyze_degeneracy",
    "check_photometric_rescue",
    "compute_geometric_fim",
    "compute_photometric_fim",
]
