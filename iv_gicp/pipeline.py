"""
IV-GICP Full Pipeline

PointCloud → [range filter] → [constant-velocity prediction] →
AdaptiveVoxelMap → IV-GICP registration →
(optional) FactorGraph → RetroactivePropagation or FORM-style update → RefinedMap

Key improvements over draft:
  - Constant-velocity motion prediction for ICP initial pose (like KISS-ICP)
  - Range filtering to remove close-range noise and far-range outliers
  - Sliding-window map: limits accumulated points to prevent O(N·logN) rebuild growth
  - AdaptiveVoxelMap.build() leaves list now properly reset between calls
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from scipy.spatial import cKDTree

from .adaptive_voxelization import AdaptiveVoxelMap
from .iv_gicp import IVGICP, Voxel4D, build_combined_covariance
from .fast_kdtree import FastKDTree
from .distribution_propagation import DistributionPropagator
from .factor_graph import PoseGraphOptimizer
from .se3_utils import se3_inverse, se3_compose
from .gpu_backend import batch_precision_matrices, get_device


def voxel_downsample(
    points: np.ndarray,
    intensities: np.ndarray,
    voxel_size: float,
) -> tuple:
    """
    Voxel-grid downsampling: select one point per voxel (first hit).
    O(N) via hash-based binning. Preserves spatial coverage while
    drastically reducing point count (e.g. 120k → 5-10k).

    This matches the preprocessing used by KISS-ICP and VGICP.
    """
    keys = (points[:, :3] / voxel_size).astype(np.int64)
    # Cantor-like hash for 3D integer keys (avoids Python dict overhead)
    hashes = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, unique_idx = np.unique(hashes, return_index=True)
    return points[unique_idx], intensities[unique_idx]


@dataclass
class OdometryResult:
    """Single frame odometry result."""
    pose: np.ndarray                # 4×4 absolute pose (world ← sensor)
    timestamp: Optional[float] = None
    num_correspondences: int = 0


@dataclass
class Trajectory:
    """Trajectory and map state."""
    poses: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    voxel_map_means: Optional[np.ndarray] = None
    voxel_map_covs: Optional[List[np.ndarray]] = None


class IVGICPPipeline:
    """
    Full IV-GICP odometry pipeline.

    Processes LiDAR scans sequentially and maintains a growing voxel map.
    Uses constant-velocity motion model for initial pose prediction.
    """

    def __init__(
        self,
        # Adaptive voxelization
        voxel_size: float = 1.0,
        entropy_threshold: float = 2.0,
        intensity_var_threshold: float = 100.0,
        min_points_per_voxel: int = 3,
        max_depth: int = 6,           # reduced from 8 for speed
        # IV-GICP registration
        alpha: float = 0.1,           # corrected from 0.01: ~range/max_I
        max_correspondence_distance: float = 2.0,
        max_iterations: int = 32,
        # Preprocessing
        min_range: float = 0.5,       # [m] remove close-range returns
        max_range: float = 80.0,      # [m] remove far-range noise
        source_voxel_size: float = 0.5,  # [m] voxel downsample source scan (0 = disabled)
        # Map management
        max_map_points: int = 100_000,  # sliding window to bound rebuild cost
        # Distribution propagation
        use_distribution_propagation: bool = True,
        # GPU acceleration
        device: str = "auto",
    ):
        self._device = get_device(device)
        self.adaptive_map = AdaptiveVoxelMap(
            voxel_size=voxel_size,
            entropy_threshold=entropy_threshold,
            intensity_var_threshold=intensity_var_threshold,
            min_points_per_voxel=min_points_per_voxel,
            max_depth=max_depth,
        )
        self.iv_gicp = IVGICP(
            alpha=alpha,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            min_voxel_size=voxel_size,
            device=device,
        )
        self.propagator = DistributionPropagator()
        self.use_distribution_propagation = use_distribution_propagation

        self.min_range = min_range
        self.max_range = max_range
        self.source_voxel_size = source_voxel_size
        self.max_map_points = max_map_points

        self.trajectory = Trajectory()
        self.current_pose = np.eye(4)
        self.map_voxels: Optional[List[tuple]] = None   # (mean, cov, mean_i, n)
        self.accumulated_points: Optional[np.ndarray] = None  # (N, 4) [x,y,z,I] world

        # Cached registration targets (rebuilt from adaptive map)
        self._target_voxels_4d: Optional[List[Voxel4D]] = None
        self._target_means_3d: Optional[np.ndarray] = None
        self._target_tree: Optional[FastKDTree] = None
        self._frame_count: int = 0
        self._rebuild_interval: int = 10  # full octree rebuild every K frames

    def _prefilter(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
    ) -> tuple:
        """
        Range filter + voxel downsampling.
        1. Remove points too close (sensor noise) or too far (unreliable returns)
        2. Voxel-grid downsample to reduce point count (e.g. 120k → 5-10k)
        Matches KISS-ICP preprocessing pipeline.
        """
        ranges = np.linalg.norm(points[:, :3], axis=1)
        mask = (ranges > self.min_range) & (ranges < self.max_range)
        pts, ints = points[mask], intensities[mask]
        if self.source_voxel_size > 0 and len(pts) > 0:
            pts, ints = voxel_downsample(pts, ints, self.source_voxel_size)
        return pts, ints

    def _predict_initial_pose(self) -> np.ndarray:
        """
        Constant-velocity motion model: predict next pose by extrapolating
        the last relative motion. This is the key trick used in KISS-ICP
        that dramatically improves ICP convergence (±30-40% ATE reduction).

        If fewer than 2 poses exist, returns the current pose.
        """
        poses = self.trajectory.poses
        if len(poses) >= 2:
            T_prev_prev = poses[-2]
            T_prev = poses[-1]
            # Relative motion: T_{k-1→k}
            T_rel = np.linalg.inv(T_prev_prev) @ T_prev
            # Predict T_{k→k+1} assuming same relative motion
            return T_prev @ T_rel
        return self.current_pose.copy()

    def _build_voxels_from_adaptive_map(self) -> None:
        """
        Convert AdaptiveVoxelMap leaves to Voxel4D for direct registration.

        This is the critical connection that allows adaptive octree statistics
        (variable-resolution covariances, per-voxel intensity variance) to be
        used directly by IV-GICP, rather than discarding them and rebuilding
        a flat grid internally.

        Each leaf's half_size is used as the effective voxel_size for σ_I²
        computation, so fine voxels (high entropy/texture regions) produce
        tighter photometric constraints.
        """
        alpha = self.iv_gicp.alpha
        # Collect all valid leaves first, then batch GPU ops
        leaf_stats = []
        for leaf in self.adaptive_map.leaves:
            s = leaf.stats
            if s is None or s.n_points < 2:
                continue
            leaf_stats.append((s, max(leaf.half_size * 2, 1e-3)))

        if not leaf_stats:
            return

        # Build batch arrays
        covs_arr   = np.array([s.cov for s, _ in leaf_stats])              # (n, 3, 3)
        var_i_arr  = np.array([s.var_intensity for s, _ in leaf_stats])    # (n,)
        vsizes_arr = np.array([vs for _, vs in leaf_stats])                 # (n,)

        # Batch precision matrices on GPU (replaces per-leaf pinv loop)
        precisions = batch_precision_matrices(
            covs_arr, var_i_arr, vsizes_arr, alpha,
            device=self._device,
        )  # (n, 4, 4)

        voxels_4d: List[Voxel4D] = []
        means_3d: List[np.ndarray] = []
        for i, (s, vs) in enumerate(leaf_stats):
            mean_4d = np.array([s.mean[0], s.mean[1], s.mean[2], alpha * s.mean_intensity])
            C = build_combined_covariance(s.cov, s.var_intensity, vs, alpha)
            voxels_4d.append(Voxel4D(mean=mean_4d, cov=C, precision=precisions[i]))
            means_3d.append(s.mean)

        if not means_3d:
            return

        means_arr = np.array(means_3d)
        tree = FastKDTree(means_arr)

        # Compute intensity gradients ∇μ_I via K-NN
        self.iv_gicp._compute_intensity_gradients(means_arr, voxels_4d, tree)

        self._target_voxels_4d = voxels_4d
        self._target_means_3d = means_arr
        self._target_tree = tree

    def process_frame(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        init_pose: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> OdometryResult:
        """
        Process one LiDAR scan. Returns absolute pose and updates internal map.

        Uses direct adaptive voxel → registration path: AdaptiveVoxelMap leaves
        are converted to Voxel4D and passed directly to IV-GICP, preserving
        variable-resolution covariances and intensity constraints.

        Args:
            points:      (N, 3+) array with at least [x, y, z] columns
            intensities: (N,) intensity values; if None, extracted from points[:,3]
            init_pose:   override for initial pose estimate (default: constant-velocity)
            timestamp:   scan timestamp (optional)
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if intensities is None:
            intensities = (
                points[:, 3] if points.shape[1] >= 4
                else np.zeros(len(points))
            )

        # Preprocessing: range filter + voxel downsampling
        points, intensities = self._prefilter(points, intensities)
        if len(points) < 10:
            pose = self.current_pose.copy()
            self.trajectory.poses.append(pose)
            self.trajectory.timestamps.append(timestamp or 0.0)
            return OdometryResult(pose=pose, timestamp=timestamp)

        # First frame: initialize map only
        if self.map_voxels is None:
            pts_world = points[:, :3].copy()
            self.accumulated_points = np.column_stack([pts_world, intensities])
            self.adaptive_map.build(self.accumulated_points, intensities)
            self.map_voxels = self.adaptive_map.get_voxels()
            self._build_voxels_from_adaptive_map()
            self._frame_count = 1
            self.trajectory.poses = [np.eye(4)]
            self.trajectory.timestamps = [timestamp or 0.0]
            return OdometryResult(pose=np.eye(4), timestamp=timestamp)

        # Initial pose: constant-velocity prediction (unless overridden)
        if init_pose is None:
            init_pose = self._predict_initial_pose()

        # Register source → adaptive voxel map (direct path)
        if self._target_voxels_4d is not None and self._target_tree is not None:
            T = self.iv_gicp.register_with_voxel_map(
                np.column_stack([points[:, :3], intensities]),
                intensities,
                self._target_voxels_4d,
                self._target_means_3d,
                self._target_tree,
                init_pose=init_pose,
            )
        else:
            # Fallback: use old register() path
            target_means = np.array([v[0] for v in self.map_voxels])
            target_intensities = np.array([v[2] for v in self.map_voxels])
            target_points = np.column_stack([target_means, target_intensities])
            source_points = np.column_stack([points[:, :3], intensities])
            T = self.iv_gicp.register(
                source_points, target_points,
                source_intensities=intensities,
                target_intensities=target_intensities,
                init_pose=init_pose,
            )

        self.current_pose = T
        self.trajectory.poses.append(self.current_pose.copy())
        self.trajectory.timestamps.append(
            timestamp or self.trajectory.timestamps[-1] + 1.0
        )

        # Update map with newly registered scan
        self._update_map(points, intensities, T)
        self._frame_count += 1

        return OdometryResult(pose=T, timestamp=timestamp)

    def _update_map(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
        T: np.ndarray,
    ) -> None:
        """
        Append new scan (transformed to world frame) to the local map and
        rebuild voxel statistics.

        Sliding window: limits total accumulated points to max_map_points
        to prevent O(N_total · log N_total) rebuild cost growth. The oldest
        points (from the earliest frames) are dropped when the cap is reached.
        This matches the local-map approach used in KISS-ICP.

        After rebuilding the adaptive octree, converts leaves to Voxel4D
        for direct use in next frame's registration.
        """
        pts_world = (T[:3, :3] @ points[:, :3].T).T + T[:3, 3]
        new_pts = np.column_stack([pts_world, intensities])

        if self.accumulated_points is not None:
            self.accumulated_points = np.vstack([self.accumulated_points, new_pts])
            # Enforce sliding window to bound rebuild cost
            if len(self.accumulated_points) > self.max_map_points:
                self.accumulated_points = self.accumulated_points[-self.max_map_points:]
        else:
            self.accumulated_points = new_pts

        self.adaptive_map.build(
            self.accumulated_points,
            self.accumulated_points[:, 3],
        )
        self.map_voxels = self.adaptive_map.get_voxels()

        # Rebuild registration targets from adaptive map
        self._build_voxels_from_adaptive_map()

    def apply_retroactive_correction(self, delta_T: np.ndarray) -> None:
        """
        Apply pose correction from factor graph via distribution propagation.
        Updates voxel (mean, covariance) pairs in O(V) instead of O(N).
        """
        if not self.use_distribution_propagation or self.map_voxels is None:
            return

        voxel_ids = list(range(len(self.map_voxels)))
        means = [v[0] for v in self.map_voxels]
        covs  = [v[1] for v in self.map_voxels]
        self.propagator.set_voxel_map(voxel_ids, means, covs)
        self.propagator.propagate(delta_T)

        new_map = self.propagator.get_voxel_map()
        self.map_voxels = [
            (new_map[i].mean, new_map[i].cov, self.map_voxels[i][2], self.map_voxels[i][3])
            for i in range(len(self.map_voxels))
        ]

    def run_factor_graph_smoothing(
        self,
        lag_size: int = 10,
        use_distribution_propagation: bool = True,
    ) -> Optional[List[np.ndarray]]:
        """
        Fixed-lag factor graph optimization followed by map update.
        Returns optimized poses (or None if fewer than 2 frames).
        """
        if len(self.trajectory.poses) < 2:
            return None

        optimizer = PoseGraphOptimizer(lag_size=lag_size)
        optimizer.set_prior(self.trajectory.poses[0])
        for i, T in enumerate(self.trajectory.poses):
            optimizer.add_pose(T)
            if i > 0:
                T_rel = se3_compose(se3_inverse(self.trajectory.poses[i - 1]), T)
                optimizer.add_odometry_factor(i - 1, i, T_rel)

        opt_poses = optimizer.optimize()
        if opt_poses is None:
            return None

        if use_distribution_propagation and self.map_voxels is not None:
            self._apply_retroactive_from_poses(opt_poses)

        self.trajectory.poses = opt_poses
        self.current_pose = opt_poses[-1]
        return opt_poses

    def _apply_retroactive_from_poses(
        self,
        optimized_poses: List[np.ndarray],
    ) -> None:
        """Apply per-frame pose corrections via distribution propagation."""
        if self.map_voxels is None or len(self.trajectory.poses) == 0:
            return

        delta_per_frame = {}
        for fid in range(min(len(self.trajectory.poses), len(optimized_poses))):
            T_old = self.trajectory.poses[fid]
            T_new = optimized_poses[fid]
            if not np.allclose(T_old, T_new, atol=1e-6):
                delta_per_frame[fid] = se3_compose(T_new, se3_inverse(T_old))

        if not delta_per_frame:
            return

        # Composite delta for all frames
        delta_composite = np.eye(4)
        for fid in sorted(delta_per_frame.keys()):
            delta_composite = se3_compose(delta_composite, delta_per_frame[fid])

        self.propagator.set_voxel_map(
            list(range(len(self.map_voxels))),
            [v[0] for v in self.map_voxels],
            [v[1] for v in self.map_voxels],
        )
        self.propagator.propagate(delta_composite)

        new_map = self.propagator.get_voxel_map()
        self.map_voxels = [
            (new_map[i].mean, new_map[i].cov, self.map_voxels[i][2], self.map_voxels[i][3])
            for i in range(len(self.map_voxels))
        ]

    def get_trajectory(self) -> Trajectory:
        return self.trajectory
