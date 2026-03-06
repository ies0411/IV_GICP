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

import time
import collections
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from scipy.spatial import cKDTree

from .adaptive_voxelization import AdaptiveVoxelMap
from .flat_voxel_map import FlatVoxelMap, AdaptiveFlatVoxelMap, LocalKeyframeVoxelMap
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
    Voxel-grid downsampling: centroid per voxel (KISS-ICP / PCL style).
    O(N log N) via sort+reduceat. More stable than first-hit because
    the centroid is the geometric center of all points in the voxel,
    not an arbitrary first point that may be an outlier.
    """
    keys = (points[:, :3] / voxel_size).astype(np.int64)
    hashes = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    order = np.argsort(hashes)
    hashes_s = hashes[order]
    pts_s = points[order]
    ints_s = intensities[order]
    _, first_occ, counts = np.unique(hashes_s, return_index=True, return_counts=True)
    pts_centroids = np.add.reduceat(pts_s, first_occ, axis=0) / counts[:, None]
    ints_centroids = np.add.reduceat(ints_s, first_occ) / counts
    return pts_centroids, ints_centroids


@dataclass
class OdometryResult:
    """Single frame odometry result."""
    pose: np.ndarray                # 4×4 absolute pose (world ← sensor)
    timestamp: Optional[float] = None
    num_correspondences: int = 0
    # Per-phase timing (ms), 0.0 if not measured
    reg_ms: float = 0.0     # ICP registration time
    map_ms: float = 0.0     # map insert + optional KDTree rebuild time


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
        intensity_range_correction: bool = False,  # I_cal = I × r² / r0² (range-invariant reflectivity)
        entropy_threshold: float = 2.0,
        intensity_var_threshold: float = 100.0,
        min_points_per_voxel: int = 3,
        max_depth: int = 6,           # reduced from 8 for speed
        # IV-GICP registration
        alpha: float = 0.1,           # corrected from 0.01: ~range/max_I
        max_correspondence_distance: float = 2.0,
        max_iterations: int = 32,
        # Adaptive threshold (KISS-ICP style sigma tracking)
        initial_threshold: float = 2.0,    # initial correspondence distance sigma
        min_motion_th: float = 0.1,        # ignore frames with motion < this (m)
        # Robust kernel
        huber_delta: float = 1.0,          # Huber threshold; 0 = disabled
        # Preprocessing
        min_range: float = 0.5,       # [m] remove close-range returns
        max_range: float = 80.0,      # [m] remove far-range noise
        source_voxel_size: float = 0.5,  # [m] voxel downsample source scan (0 = disabled)
        # Map management
        max_map_points: int = 100_000,  # sliding window to bound rebuild cost
        max_map_frames: Optional[int] = None,  # override frame window (None = auto from max_map_points)
        adaptive_voxelization: bool = True,  # C1: entropy-based coarse/fine split (slower but better for tunnels)
        # Multi-scale registration
        # coarse_voxel_size > 0 enables two-level GICP:
        #   1. Coarse map (coarse_voxel_size): initial convergence, large search radius
        #   2. Fine map (voxel_size): precise refinement starting from coarse result
        # Theoretical backing: multi-scale ICP reduces susceptibility to local minima
        # by providing a wide convergence basin at the coarse level, then refining
        # with fine voxels. See Chetverikov et al. 2002, Magnusson et al. 2009.
        coarse_voxel_size: float = 0.0,   # 0 = disabled; e.g. 3.0 for outdoor
        coarse_iterations: int = 10,       # GN iterations for coarse level
        coarse_max_corr: float = 0.0,      # 0 = auto (3 × coarse_voxel_size)
        # FORM-style fixed-lag window smoothing (Potokar et al. 2025)
        # window_size > 1 enables joint pose graph optimization over the K most
        # recent frames using ICP Hessians as factor information matrices.
        # Faster than GTSAM: uses our GN solver on a tiny (6K × 6K) linear system.
        window_size: int = 1,   # 1 = disabled; typically 5-15 for real-time use
        # Distribution propagation
        use_distribution_propagation: bool = True,
        # GPU acceleration
        device: str = "auto",
    ):
        self._device = get_device(device)
        # Keep AdaptiveVoxelMap for offline analysis / ablation study.
        # Real-time odometry now uses FlatVoxelMap (O(N_frame) per update).
        self.adaptive_map = AdaptiveVoxelMap(
            voxel_size=voxel_size,
            entropy_threshold=entropy_threshold,
            intensity_var_threshold=intensity_var_threshold,
            min_points_per_voxel=min_points_per_voxel,
            max_depth=max_depth,
        )
        # AdaptiveFlatVoxelMap (C1):
        #   C1: world-frame merged voxel map with entropy-based adaptive resolution.
        #       Coarse voxels for flat/simple regions, fine for complex geometry.
        #       Combined splitting score S = H_geo + λ·H_int > τ_split.
        #   World-frame merging (Welford) gives accurate multi-frame covariances
        #   essential for reliable GICP in degenerate scenes (tunnels, corridors).
        # LocalKeyframeVoxelMap (C3) is kept for retroactive correction with a
        # factor graph smoother (not yet integrated into the real-time odometry path).
        _map_frames = max_map_frames if max_map_frames is not None else max(max_map_points // 5000, 20)
        if adaptive_voxelization:
            # C1: entropy-based dual coarse/fine map
            self.flat_map = AdaptiveFlatVoxelMap(
                base_voxel_size=voxel_size,
                min_points_coarse=min_points_per_voxel,
                max_frames=_map_frames,
                entropy_threshold=entropy_threshold,
                lambda_intensity=0.1,
            )
        else:
            # Plain FlatVoxelMap (faster for large outdoor scenes; ablation baseline)
            self.flat_map = FlatVoxelMap(voxel_size, min_points_per_voxel, _map_frames)
        self.iv_gicp = IVGICP(
            alpha=alpha,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            min_voxel_size=voxel_size,
            huber_delta=huber_delta,
            source_sigma=source_voxel_size / 2.0,
            device=device,
        )

        # Multi-scale: coarse map + ICP for better convergence basin
        self._coarse_map: Optional[FlatVoxelMap] = None
        self._coarse_iv_gicp: Optional[IVGICP] = None
        self._coarse_target_voxels_4d: Optional[List[Voxel4D]] = None
        self._coarse_target_means_3d: Optional[np.ndarray] = None
        self._coarse_target_tree: Optional[FastKDTree] = None
        self.coarse_voxel_size = coarse_voxel_size
        _coarse_max_corr = coarse_max_corr if coarse_max_corr > 0 else 3.0 * coarse_voxel_size
        self._coarse_max_corr = _coarse_max_corr
        if coarse_voxel_size > 0:
            _coarse_map_frames = max(3, _map_frames // 3)
            self._coarse_map = FlatVoxelMap(coarse_voxel_size, 2, _coarse_map_frames)
            self._coarse_iv_gicp = IVGICP(
                alpha=0.0,
                max_correspondence_distance=_coarse_max_corr,
                max_iterations=coarse_iterations,
                min_voxel_size=coarse_voxel_size,
                huber_delta=0.0,
                source_sigma=source_voxel_size / 2.0,
                device=device,
            )

        self.propagator = DistributionPropagator()
        self.use_distribution_propagation = use_distribution_propagation

        # FORM-style window smoothing buffer.
        # Each entry: (T_abs, T_rel, H_icp) — absolute pose, relative pose, ICP Hessian.
        # FORM key insight: ICP Hessian H encodes which DOFs are well-constrained
        # (high eigenvalue = informative direction). Joint optimization exploits this
        # to improve estimates for degenerate frames from adjacent well-constrained ones.
        # Faster than GTSAM: our GN solver handles 6K×6K system in <1ms for K≤15.
        self._window_size = window_size
        self._window_buffer: collections.deque = collections.deque(maxlen=window_size)

        self.min_range = min_range
        self.max_range = max_range
        self.source_voxel_size = source_voxel_size
        self.max_map_points = max_map_points
        self.intensity_range_correction = intensity_range_correction

        # Adaptive threshold state (KISS-ICP-style sigma tracking).
        # _adaptive_sigma = σ (median correspondence distance from last good frame).
        # adaptive_corr = 3σ (3-sigma rule).
        # Initialized to max_corr_dist/3 so the first registration uses the full
        # search range, then sigma naturally converges to actual quality.
        self._adaptive_sigma = max_correspondence_distance / 3.0
        self._initial_threshold = initial_threshold
        self._min_motion_th = min_motion_th

        self.trajectory = Trajectory()
        self.current_pose = np.eye(4)
        self.map_voxels: Optional[List[tuple]] = None   # legacy (for C3 propagation)
        self.accumulated_points: Optional[np.ndarray] = None  # legacy (for ablation)

        # Cached registration targets (rebuilt from flat map)
        self._target_voxels_4d: Optional[List[Voxel4D]] = None
        self._target_means_3d: Optional[np.ndarray] = None
        self._target_tree: Optional[FastKDTree] = None
        self._frame_count: int = 0
        # Rebuild KDTree + Voxel4D only every _kdtree_interval frames.
        # Between rebuilds: use cached tree (slightly stale but fast).
        self._kdtree_interval: int = 3

    def _prefilter(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
    ) -> tuple:
        """
        Range filter + voxel downsampling + intensity normalization.
        1. Remove points too close (sensor noise) or too far (unreliable returns)
        2. Voxel-grid downsample to reduce point count (e.g. 120k → 5-10k)
        3. Normalize intensities to [0, 1] if they exceed 1.0 (e.g. raw [0-255] LiDAR)
           This ensures omega_I scaling is consistent across sensors.
           KITTI: already [0,1] → no-op.  Hilti Pandar: [0-200] → normalized.
        Matches KISS-ICP preprocessing pipeline.
        """
        ranges = np.linalg.norm(points[:, :3], axis=1)
        mask = (ranges > self.min_range) & (ranges < self.max_range)
        pts, ints = points[mask], intensities[mask]
        ranges_m = ranges[mask]
        # Range-based intensity calibration: I_cal = I × r² (range-invariant reflectivity).
        # Physical model: returned power ∝ reflectivity / r² → multiply by r² to recover
        # true reflectivity. r0=5m reference range keeps I_cal in a similar scale to I.
        # Effect: far walls appear brighter (less distance attenuation bias),
        # near walls appear similar. Creates texture variation in otherwise uniform tunnels.
        if self.intensity_range_correction and len(ints) > 0:
            r0 = 5.0  # reference range [m]
            ints = ints * (ranges_m / r0) ** 2
        if self.source_voxel_size > 0 and len(pts) > 0:
            pts, ints = voxel_downsample(pts, ints, self.source_voxel_size)
        # Intensity normalization: auto-scale to [0, 1] when sensor reports raw counts
        if len(ints) > 0:
            p99 = float(np.percentile(ints, 99))
            if p99 > 1.0:
                ints = ints / p99
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
        Convert map leaves to Voxel4D for direct registration.

        Works with both AdaptiveVoxelMap (offline) and FlatVoxelMap (real-time).
        The default real-time path uses FlatVoxelMap.leaves (same interface).

        Each leaf's half_size is the effective voxel_size for σ_I² computation.
        FlatVoxelMap uses uniform half_size = voxel_size / 2 for all leaves.
        """
        alpha = self.iv_gicp.alpha
        # Use FlatVoxelMap for real-time odometry (fast incremental updates).
        leaf_stats = []
        for leaf in self.flat_map.leaves:
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
        n_counts   = np.array([s.n_points for s, _ in leaf_stats])         # (n,)

        # Batch precision matrices on GPU (replaces per-leaf pinv loop).
        # source_sigma = source_voxel_size / 2 adds a source-point noise floor
        # so Omega_max ≤ 1/source_sigma² (prevents extreme precision from thin voxels).
        # n_counts enables count-weighted regularization: sparse voxels (few points)
        # get large isotropic regularization (≈ point-to-point ICP), dense voxels
        # get accurate GICP covariance. Bridges the two extremes automatically.
        precisions = batch_precision_matrices(
            covs_arr, var_i_arr, vsizes_arr, alpha,
            source_sigma=self.source_voxel_size / 2.0,
            n_counts=n_counts,
            device=self._device,
        )  # (n, 4, 4)

        # Build Voxel4D list. Note: Voxel4D.cov is NOT used in the GN hot-path
        # (_gauss_newton_vectorized uses only .precision and .mean). Omitting cov
        # eliminates n × np.eye(4) allocations (~4μs each → 100ms+ for 32k voxels).
        n = len(leaf_stats)
        means_arr_4d = np.column_stack([
            np.array([s.mean for s, _ in leaf_stats]),                    # (n, 3)
            alpha * np.array([s.mean_intensity for s, _ in leaf_stats]),  # (n,)
        ])  # (n, 4)
        means_3d = [s.mean for s, _ in leaf_stats]
        voxels_4d = [
            Voxel4D(mean=means_arr_4d[i], precision=precisions[i])
            for i in range(n)
        ]

        if not means_3d:
            return

        means_arr = np.array(means_3d)
        tree = FastKDTree(means_arr)

        # Compute intensity gradients ∇μ_I via K-NN
        self.iv_gicp._compute_intensity_gradients(means_arr, voxels_4d, tree)

        self._target_voxels_4d = voxels_4d
        self._target_means_3d = means_arr
        self._target_tree = tree

    def _build_coarse_voxels(self) -> None:
        """Build coarse-level Voxel4D targets from coarse FlatVoxelMap."""
        leaf_stats = []
        for leaf in self._coarse_map.leaves:
            s = leaf.stats
            if s is None or s.n_points < 2:
                continue
            leaf_stats.append((s, max(leaf.half_size * 2, 1e-3)))

        if not leaf_stats:
            return

        covs_arr  = np.array([s.cov for s, _ in leaf_stats])
        var_i_arr = np.array([s.var_intensity for s, _ in leaf_stats])
        vsizes_arr = np.array([vs for _, vs in leaf_stats])
        n_counts  = np.array([s.n_points for s, _ in leaf_stats])

        precisions = batch_precision_matrices(
            covs_arr, var_i_arr, vsizes_arr, 0.0,
            source_sigma=self.source_voxel_size / 2.0,
            n_counts=n_counts,
            device=self._device,
        )

        n = len(leaf_stats)
        means_arr_4d = np.column_stack([
            np.array([s.mean for s, _ in leaf_stats]),
            np.zeros(n),
        ])
        means_3d = [s.mean for s, _ in leaf_stats]
        voxels_4d = [
            Voxel4D(mean=means_arr_4d[i], precision=precisions[i])
            for i in range(n)
        ]

        if not means_3d:
            return

        means_arr = np.array(means_3d)
        tree = FastKDTree(means_arr)
        self._coarse_iv_gicp._compute_intensity_gradients(means_arr, voxels_4d, tree)

        self._coarse_target_voxels_4d = voxels_4d
        self._coarse_target_means_3d = means_arr
        self._coarse_target_tree = tree

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

        # First frame: initialize map only (no registration target yet)
        if self._frame_count == 0:
            self.flat_map.insert_frame(points[:, :3], intensities, frame_id=0)
            self._build_voxels_from_adaptive_map()
            if self._coarse_map is not None:
                self._coarse_map.insert_frame(points[:, :3], intensities, frame_id=0)
                self._build_coarse_voxels()
            self.map_voxels = []  # mark as initialized
            self._frame_count = 1
            self.trajectory.poses = [np.eye(4)]
            self.trajectory.timestamps = [timestamp or 0.0]
            return OdometryResult(pose=np.eye(4), timestamp=timestamp)

        # Initial pose: constant-velocity prediction (unless overridden)
        if init_pose is None:
            init_pose = self._predict_initial_pose()

        # Adaptive correspondence distance (KISS-ICP style):
        # Use 3σ rule where σ is updated from actual registration quality.
        # This matches KISS-ICP's adaptive threshold mechanism:
        #   threshold = max(3 × pred_motion,  3 × σ_est)
        # σ_est is updated post-registration from median correspondence distance.
        pred_motion = np.linalg.norm(init_pose[:3, 3] - self.current_pose[:3, 3])
        adaptive_corr = max(3.0 * pred_motion, 3.0 * self._adaptive_sigma)
        adaptive_corr = min(adaptive_corr, self.iv_gicp.max_corr_dist)

        # Multi-scale: run coarse GICP first to improve initial pose estimate.
        # Coarse map has larger voxels → wider convergence basin → fewer local minima.
        # Result feeds as init_pose into the fine-level registration.
        if (self._coarse_map is not None
                and self._coarse_target_tree is not None
                and self._coarse_target_voxels_4d is not None):
            T_coarse, _ = self._coarse_iv_gicp.register_with_voxel_map(
                np.column_stack([points[:, :3], intensities]),
                intensities,
                self._coarse_target_voxels_4d,
                self._coarse_target_means_3d,
                self._coarse_target_tree,
                init_pose=init_pose,
                max_corr_dist=self._coarse_max_corr,
            )
            init_pose = T_coarse

        # Register source → adaptive voxel map (direct path)
        reg_info = {"n_correspondences": 0, "converged": False}
        _t_reg = time.perf_counter()
        if self._target_voxels_4d is not None and self._target_tree is not None:
            T, reg_info = self.iv_gicp.register_with_voxel_map(
                np.column_stack([points[:, :3], intensities]),
                intensities,
                self._target_voxels_4d,
                self._target_means_3d,
                self._target_tree,
                init_pose=init_pose,
                max_corr_dist=adaptive_corr,
            )
        else:
            # Fallback: use old register() path
            target_means = np.array([v[0] for v in self.map_voxels])
            if len(target_means) == 0:
                # Map not ready yet (e.g. all singletons filtered) — skip registration
                T = init_pose.copy() if init_pose is not None else self.current_pose.copy()
                reg_info = {"n_correspondences": 0, "converged": False}
            else:
                target_intensities = np.array([v[2] for v in self.map_voxels])
                target_points = np.column_stack([target_means, target_intensities])
                source_points = np.column_stack([points[:, :3], intensities])
                T = self.iv_gicp.register(
                    source_points, target_points,
                    source_intensities=intensities,
                    target_intensities=target_intensities,
                    init_pose=init_pose,
                )
                reg_info = {"n_correspondences": 999, "converged": True}
        reg_ms = (time.perf_counter() - _t_reg) * 1000

        # Quality-based sanity check (replaces fixed-threshold check).
        # Reject if: (1) virtually no correspondences found, meaning ICP had no data,
        # OR (2) result is physically implausible (> 2 × max_corr_dist from prediction).
        # Skip map update on poor frames to prevent map corruption (cascade failure).
        n_corr = reg_info.get("n_correspondences", 0)
        result_dist = np.linalg.norm(T[:3, 3] - init_pose[:3, 3])
        poor_registration = (n_corr < 10) or (result_dist > 2.0 * self.iv_gicp.max_corr_dist)
        if poor_registration and self._frame_count > 1:
            T = init_pose.copy()

        # FORM-style fixed-lag window smoothing (Potokar et al. 2025, arXiv 2510.09966).
        # Joint GN optimization over the K-pose sliding window using ICP Hessians as
        # factor information matrices. Fills in degenerate DOFs (low H eigenvalue)
        # from adjacent well-constrained frames. <0.5ms for K≤15.
        if self._window_size > 1 and not poor_registration:
            H_icp = reg_info.get("hessian")
            if H_icp is not None:
                T_rel_cur = se3_compose(se3_inverse(self.current_pose), T)
                T = self._window_smooth(T, T_rel_cur, H_icp, self._frame_count)

        # Update adaptive sigma from accepted registration quality.
        # sigma = median correspondence distance between accepted-pose source and target.
        # This directly measures map-to-scan alignment quality for the next frame's threshold.
        # Using correspondence distances (not velocity deviation) is more stable because
        # it measures actual map quality independent of motion estimation errors.
        if self._target_tree is not None and len(points) > 5:
            pts_world = (T[:3, :3] @ points[:, :3].T).T + T[:3, 3]
            dists, _ = self._target_tree.query(pts_world, k=1)
            inlier_mask = dists < adaptive_corr
            if np.sum(inlier_mask) > 5:
                sigma_new = max(float(np.median(dists[inlier_mask])), self._min_motion_th)
                self._adaptive_sigma = sigma_new

        self.current_pose = T
        self.trajectory.poses.append(self.current_pose.copy())
        self.trajectory.timestamps.append(
            timestamp or self.trajectory.timestamps[-1] + 1.0
        )

        # Only update the map when registration quality was sufficient.
        # Skipping map update on poor frames prevents corrupted points from
        # accumulating and causing cascade failures on subsequent frames.
        _t_map = time.perf_counter()
        if not poor_registration:
            self._update_map(points, intensities, T, self._frame_count)
        map_ms = (time.perf_counter() - _t_map) * 1000
        self._frame_count += 1

        return OdometryResult(
            pose=T, timestamp=timestamp,
            num_correspondences=n_corr,
            reg_ms=reg_ms, map_ms=map_ms,
        )

    def _update_map(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
        T: np.ndarray,
        frame_id: int,
    ) -> None:
        """
        Incremental map update: O(N_frame) per call.

        Transforms source points to world frame and inserts into the voxel map.
        World-frame merging (Welford) gives robust multi-frame covariances for GICP.
        Sliding window eviction keeps map bounded at max_frames keyframes.
        """
        pts_world = (T[:3, :3] @ points[:, :3].T).T + T[:3, 3]
        self.flat_map.insert_frame(pts_world, intensities, frame_id)

        # Sliding window: evict voxels not observed in the last max_frames frames.
        # Evict every 10 frames to amortize O(V) dict traversal over large outdoor maps.
        if self._frame_count % 10 == 0:
            evict_before = self._frame_count - self.flat_map.max_frames
            if evict_before > 0:
                self.flat_map.evict_before(evict_before)

        # Rebuild Voxel4D targets at controlled interval (not every frame).
        # The map stats are updated every frame (Welford); KDTree is cached.
        if (self._frame_count % self._kdtree_interval == 0
                or self._target_tree is None):
            self._build_voxels_from_adaptive_map()

        # Update coarse map (same sliding window, separate eviction)
        if self._coarse_map is not None:
            self._coarse_map.insert_frame(pts_world, intensities, self._frame_count)
            if self._frame_count % 10 == 0:
                evict_before_c = self._frame_count - self._coarse_map.max_frames
                if evict_before_c > 0:
                    self._coarse_map.evict_before(evict_before_c)
            if (self._frame_count % self._kdtree_interval == 0
                    or self._coarse_target_tree is None):
                self._build_coarse_voxels()

    def _window_smooth(
        self,
        T_new: np.ndarray,
        T_rel: np.ndarray,
        H: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        """
        FORM-style fixed-lag window smoothing with C3 retroactive map update.

        Step 1 — Pose smoothing (FORM, Potokar et al. 2025, arXiv 2510.09966):
            Jointly optimizes K poses using ICP Hessians as information matrices.
            High eigenvalue = well-constrained DOF; low = degenerate.
            Adjacent well-constrained frames repair degenerate DOFs.

        Step 2 — Map update (C3, O(V)):
            After smoothing, calls LocalKeyframeVoxelMap.update_poses() to
            transform stored local voxel distributions to corrected world frame.
            Cost: O(V) matrix multiply per voxel — no raw-point reprocessing.
            This is the key C3 advantage over FORM's O(N×W) reconstruction.

        Args:
            T_new:    raw ICP pose for current frame (4×4 world←sensor)
            T_rel:    relative motion from prev pose (T_prev^{-1} @ T_new)
            H:        ICP Hessian (6×6), approximates Fisher info of this measurement
            frame_id: current frame index (for C3 map update bookkeeping)

        Returns:
            Optimized current pose (4×4).
        """
        self._window_buffer.append((T_new.copy(), T_rel.copy(), H.copy(), frame_id))
        K = len(self._window_buffer)
        if K < 2:
            return T_new

        buf = list(self._window_buffer)

        optimizer = PoseGraphOptimizer(
            lag_size=K,
            max_iterations=3,
            convergence_threshold=1e-6,
            prior_weight=1e4,
        )
        optimizer.set_prior(buf[0][0])
        for T_abs, _, _, _ in buf:
            optimizer.add_pose(T_abs)

        for i in range(1, K):
            T_rel_i = buf[i][1]
            H_i     = buf[i][2]
            H_sym = (H_i + H_i.T) * 0.5
            min_eig = float(np.linalg.eigvalsh(H_sym)[0])
            if min_eig < 1e-4:
                H_sym = H_sym + (1e-4 - min_eig) * np.eye(6)
            optimizer.add_odometry_factor(i - 1, i, T_rel_i, omega=H_sym)

        opt_poses = optimizer.optimize()
        if opt_poses is None or len(opt_poses) < K:
            return T_new

        # Step 1: retroactively correct trajectory poses.
        n_traj = len(self.trajectory.poses)
        for k in range(K - 1):
            traj_idx = n_traj - (K - 1) + k
            if 0 <= traj_idx < n_traj:
                self.trajectory.poses[traj_idx] = opt_poses[k]

        # NOTE: Step 2 (C3 map update via apply_delta_transform) is intentionally
        # omitted from the per-frame sliding window path. The sliding window overlaps
        # mean each frame appears in ~window_size consecutive windows, causing its
        # correction to be applied ~window_size times → cumulative map drift.
        # C3 map update is only safe when triggered once per loop-closure event,
        # not per-frame. See docs/c3.md for detailed analysis.

        # Update buffer with corrected poses for next frame's T_rel computation.
        new_buf = [(opt_poses[k], buf[k][1], buf[k][2], buf[k][3]) for k in range(K)]
        self._window_buffer.clear()
        for entry in new_buf:
            self._window_buffer.append(entry)

        return opt_poses[-1]

    def update_map_poses(self, frame_pose_dict: dict) -> None:
        """
        C3: retroactive map correction when pose estimates improve.

        Args:
            frame_pose_dict: {frame_id: T_world_sensor (4×4)} for updated frames.

        After calling this, flat_map.leaves automatically uses the corrected poses
        on next access — O(V) cost, no raw-point reprocessing.
        This is the fundamental C3 advantage over FORM's O(N×W) reconstruction.
        """
        if hasattr(self.flat_map, "update_poses"):
            self.flat_map.update_poses(frame_pose_dict)
        # Force KDTree + Voxel4D rebuild at next frame
        self._target_tree = None

    def apply_retroactive_correction(self, delta_T: np.ndarray) -> None:
        """
        Apply pose correction from factor graph via distribution propagation.
        Updates voxel (mean, covariance) pairs in O(V) instead of O(N).
        """
        if not self.use_distribution_propagation or self._frame_count == 0:
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
