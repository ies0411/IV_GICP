"""
IV-GICP: Information-Theoretic Adaptive Voxelization and Intensity-Augmented GICP
with Retroactive Map Distribution Refinement.

Modules:
- adaptive_voxelization: Entropy-based octree voxelization
- iv_gicp: Geo-photometric probabilistic registration
- distribution_propagation: Retroactive map refinement (vs FORM)
- factor_graph: Factor graph optimization (fixed-lag smoothing)
- form_map: FORM-style point-wise map update (baseline)
- map_refinement: FORM vs Distribution Propagation benchmark
- pipeline: Full odometry pipeline
"""

from .iv_gicp import IVGICP
from .distribution_propagation import DistributionPropagator, VoxelDistribution
from .factor_graph import PoseGraphOptimizer, FormMapUpdater, OdometryFactor, PriorFactor
from .form_map import FormMap
from .map_refinement import MapRefinementBenchmark, RefinementResult
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
    "DistributionPropagator",
    "VoxelDistribution",
    "PoseGraphOptimizer",
    "FormMapUpdater",
    "OdometryFactor",
    "PriorFactor",
    "FormMap",
    "MapRefinementBenchmark",
    "RefinementResult",
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
