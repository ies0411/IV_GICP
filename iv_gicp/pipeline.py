"""
IV-GICP Full Pipeline

PointCloud → [range filter] → [constant-velocity prediction] →
FlatVoxelMap/AdaptiveFlatVoxelMap (or C++ VoxelMap) → IV-GICP registration →
(optional) FactorGraph → RetroactivePropagation or FORM-style update → RefinedMap

Key improvements over draft:
  - Constant-velocity motion prediction for ICP initial pose (like KISS-ICP)
  - Range filtering to remove close-range noise and far-range outliers
  - Sliding-window map: limits accumulated points to prevent O(N·logN) rebuild growth
"""

import time
import collections
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from scipy.spatial import cKDTree

from .flat_voxel_map import FlatVoxelMap, AdaptiveFlatVoxelMap
from .iv_gicp import IVGICP, Voxel4D, build_combined_covariance
from .fast_kdtree import FastKDTree
from .distribution_propagation import DistributionPropagator
from .factor_graph import PoseGraphOptimizer
from .se3_utils import se3_inverse, se3_compose
from .gpu_backend import batch_precision_matrices, get_device

try:
    from .cpp import iv_gicp_map as _cpp_map

    _CPP_MAP_AVAILABLE = True
except ImportError:
    _cpp_map = None
    _CPP_MAP_AVAILABLE = False

try:
    from .cpp import iv_gicp_core as _cpp_core
    _VOXEL_DOWNSPAMPLE_CPP = getattr(_cpp_core, "voxel_downsample", None)
except ImportError:
    _VOXEL_DOWNSPAMPLE_CPP = None


def _voxel_downsample_python(
    points: np.ndarray,
    intensities: np.ndarray,
    voxel_size: float,
) -> tuple:
    """Python fallback: voxel-grid downsampling (centroid per voxel)."""
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


def voxel_downsample(
    points: np.ndarray,
    intensities: np.ndarray,
    voxel_size: float,
) -> tuple:
    """
    Voxel-grid downsampling: centroid per voxel (KISS-ICP / PCL style).
    Uses C++ implementation from iv_gicp_core if available, else Python fallback.
    """
    if _VOXEL_DOWNSPAMPLE_CPP is not None:
        pts = np.asarray(points[:, :3], dtype=np.float64, order="C")
        ints = np.asarray(intensities, dtype=np.float64, order="C")
        out_pts, out_ints = _VOXEL_DOWNSPAMPLE_CPP(pts, ints, float(voxel_size))
        return np.asarray(out_pts), np.asarray(out_ints)
    return _voxel_downsample_python(points, intensities, voxel_size)


@dataclass
class OdometryResult:
    """Single frame odometry result."""

    pose: np.ndarray  # 4×4 absolute pose (world ← sensor)
    timestamp: Optional[float] = None
    num_correspondences: int = 0
    # Per-phase timing (ms), 0.0 if not measured
    reg_ms: float = 0.0  # ICP registration time
    map_ms: float = 0.0  # map insert + optional KDTree rebuild time


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
        max_depth: int = 6,  # reduced from 8 for speed
        # IV-GICP registration
        alpha: float = 0.1,  # corrected from 0.01: ~range/max_I
        max_correspondence_distance: float = 2.0,
        max_iterations: int = 32,
        # Adaptive threshold (KISS-ICP style sigma tracking)
        initial_threshold: float = 2.0,  # initial correspondence distance sigma
        min_motion_th: float = 0.1,  # ignore frames with motion < this (m)
        # Robust kernel
        huber_delta: float = 1.0,  # Huber threshold; 0 = disabled
        # Preprocessing
        min_range: float = 0.5,  # [m] remove close-range returns
        max_range: float = 80.0,  # [m] remove far-range noise
        source_voxel_size: float = 0.5,  # [m] voxel downsample source scan (0 = disabled)
        # Map management
        max_map_points: int = 100_000,  # sliding window to bound rebuild cost
        max_map_frames: Optional[int] = None,  # override frame window (None = auto from max_map_points)
        map_radius: Optional[float] = None,  # [m] spatial eviction radius (None = age-based eviction)
        adaptive_voxelization: bool = True,  # C1: entropy-based coarse/fine split (slower but better for tunnels)
        adaptive_max_depth: int = 2,  # C1: subdivision levels (1=coarse+fine, 2=coarse+mid+fine, …)
        min_eigenvalue_ratio: float = 0.0,  # C1-only ablation: clamp leaf cov λ_min >= η·λ_max (e.g. 0.01); 0 = off
        max_condition_number: float = 0.0,  # C1 보수적 분할: fine leaf κ > 이 값이면 coarse만 사용 (0 = 비활성, 예: 50~200)
        # Multi-scale registration
        # coarse_voxel_size > 0 enables two-level GICP:
        #   1. Coarse map (coarse_voxel_size): initial convergence, large search radius
        #   2. Fine map (voxel_size): precise refinement starting from coarse result
        # Theoretical backing: multi-scale ICP reduces susceptibility to local minima
        # by providing a wide convergence basin at the coarse level, then refining
        # with fine voxels. See Chetverikov et al. 2002, Magnusson et al. 2009.
        coarse_voxel_size: float = 0.0,  # 0 = disabled; e.g. 3.0 for outdoor
        coarse_iterations: int = 10,  # GN iterations for coarse level
        coarse_max_corr: float = 0.0,  # 0 = auto (3 × coarse_voxel_size)
        # FORM-style fixed-lag window smoothing (Potokar et al. 2025)
        # window_size > 1 enables joint pose graph optimization over the K most
        # recent frames using ICP Hessians as factor information matrices.
        # Faster than GTSAM: uses our GN solver on a tiny (6K × 6K) linear system.
        window_size: int = 1,  # 1 = disabled; typically 5-15 for real-time use
        # C2: intensity scale from data so C2 doesn't hurt in uniform-intensity scenes (no fallback to GICP)
        use_conditional_intensity: bool = False,  # if True: use geometry-only when mean(|∇I|) < threshold
        intensity_gradient_threshold: float = 1e-6,
        auto_alpha_from_intensity: bool = True,  # alpha = clip(robust_std(I), 1e-6, 1.0) from first frame when alpha>0
        # C2 auto_alpha (Theorem 1): κ high → degenerate → use intensity; κ low → geometry-only
        auto_alpha: bool = True,  # effective_alpha = alpha_max * sigmoid((κ - κ_thresh) / scale); default on
        kappa_threshold: float = 10.0,  # κ above this → increase alpha (tunnel/degenerate); lowered from 15 for earlier intensity use
        kappa_scale: float = 7.0,  # sigmoid steepness (gentler so alpha rises in mid-κ)
        alpha_floor_ratio: float = 0.2,  # when κ > threshold, effective_alpha >= alpha_max * this (avoid too-weak intensity in degenerate)
        # Distribution propagation
        use_distribution_propagation: bool = True,
        # C1: FIM-based per-voxel weighting (Python GN path only)
        use_fim_weight: bool = False,
        # C3: entropy-consistent geo/intensity balance (per-voxel omega_I scale; Python map path only)
        use_entropy_alpha: bool = False,
        entropy_scale_c: float = 0.5,  # scale = 1 + c * (h_geo - median(h_geo)), clipped [0.4, 2.5]; was 0.3 for stronger C3
        # A.1 Well-posed polish: when κ_geo < threshold, refine with geometry-only 1–2 GN steps
        well_posed_polish: bool = False,
        well_posed_kappa_threshold: float = 100.0,
        well_posed_max_iters: int = 2,
        # GPU acceleration
        device: str = "auto",
    ):
        # Merge config from YAML (if present) so callers can rely on config/ for defaults
        try:
            from .config_loader import get_pipeline_config
            _file_cfg = get_pipeline_config()
        except Exception:
            _file_cfg = {}
        _opts = dict(
            voxel_size=voxel_size,
            intensity_range_correction=intensity_range_correction,
            entropy_threshold=entropy_threshold,
            intensity_var_threshold=intensity_var_threshold,
            min_points_per_voxel=min_points_per_voxel,
            max_depth=max_depth,
            alpha=alpha,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            initial_threshold=initial_threshold,
            min_motion_th=min_motion_th,
            huber_delta=huber_delta,
            min_range=min_range,
            max_range=max_range,
            source_voxel_size=source_voxel_size,
            max_map_points=max_map_points,
            max_map_frames=max_map_frames,
            map_radius=map_radius,
            adaptive_voxelization=adaptive_voxelization,
            adaptive_max_depth=adaptive_max_depth,
            min_eigenvalue_ratio=min_eigenvalue_ratio,
            max_condition_number=max_condition_number,
            coarse_voxel_size=coarse_voxel_size,
            coarse_iterations=coarse_iterations,
            coarse_max_corr=coarse_max_corr,
            window_size=window_size,
            use_conditional_intensity=use_conditional_intensity,
            intensity_gradient_threshold=intensity_gradient_threshold,
            auto_alpha_from_intensity=auto_alpha_from_intensity,
            auto_alpha=auto_alpha,
            kappa_threshold=kappa_threshold,
            kappa_scale=kappa_scale,
            alpha_floor_ratio=alpha_floor_ratio,
            use_distribution_propagation=use_distribution_propagation,
            use_fim_weight=use_fim_weight,
            use_entropy_alpha=use_entropy_alpha,
            entropy_scale_c=entropy_scale_c,
            well_posed_polish=well_posed_polish,
            well_posed_kappa_threshold=well_posed_kappa_threshold,
            well_posed_max_iters=well_posed_max_iters,
            use_degeneracy_aware_intensity_weight=False,
            degeneracy_kappa_threshold=10.0,
            use_coarse_for_convergence=False,
            kdtree_interval=3,
            device=device,
        )
        _opts = {**_opts, **_file_cfg}
        voxel_size = _opts["voxel_size"]
        intensity_range_correction = _opts["intensity_range_correction"]
        entropy_threshold = _opts["entropy_threshold"]
        intensity_var_threshold = _opts["intensity_var_threshold"]
        min_points_per_voxel = _opts["min_points_per_voxel"]
        max_depth = _opts["max_depth"]
        alpha = _opts["alpha"]
        max_correspondence_distance = _opts["max_correspondence_distance"]
        max_iterations = _opts["max_iterations"]
        initial_threshold = _opts["initial_threshold"]
        min_motion_th = _opts["min_motion_th"]
        huber_delta = _opts["huber_delta"]
        min_range = _opts["min_range"]
        max_range = _opts["max_range"]
        source_voxel_size = _opts["source_voxel_size"]
        max_map_points = _opts["max_map_points"]
        max_map_frames = _opts["max_map_frames"]
        map_radius = _opts["map_radius"]
        adaptive_voxelization = _opts["adaptive_voxelization"]
        adaptive_max_depth = _opts["adaptive_max_depth"]
        min_eigenvalue_ratio = _opts["min_eigenvalue_ratio"]
        max_condition_number = _opts["max_condition_number"]
        coarse_voxel_size = _opts["coarse_voxel_size"]
        coarse_iterations = _opts["coarse_iterations"]
        coarse_max_corr = _opts["coarse_max_corr"]
        window_size = _opts["window_size"]
        use_conditional_intensity = _opts["use_conditional_intensity"]
        intensity_gradient_threshold = _opts["intensity_gradient_threshold"]
        auto_alpha_from_intensity = _opts["auto_alpha_from_intensity"]
        auto_alpha = _opts["auto_alpha"]
        kappa_threshold = _opts["kappa_threshold"]
        kappa_scale = _opts["kappa_scale"]
        alpha_floor_ratio = _opts["alpha_floor_ratio"]
        use_distribution_propagation = _opts["use_distribution_propagation"]
        use_fim_weight = _opts["use_fim_weight"]
        use_entropy_alpha = _opts["use_entropy_alpha"]
        entropy_scale_c = _opts["entropy_scale_c"]
        well_posed_polish = _opts.get("well_posed_polish", False)
        well_posed_kappa_threshold = float(_opts.get("well_posed_kappa_threshold", 100.0))
        well_posed_max_iters = int(_opts.get("well_posed_max_iters", 2))
        use_degeneracy_aware_intensity_weight = _opts.get("use_degeneracy_aware_intensity_weight", False)
        degeneracy_kappa_threshold = float(_opts.get("degeneracy_kappa_threshold", 10.0))
        use_coarse_for_convergence = _opts.get("use_coarse_for_convergence", False)
        kdtree_interval = int(_opts.get("kdtree_interval", 3))
        device = _opts["device"]
        if use_coarse_for_convergence and coarse_voxel_size <= 0:
            coarse_voxel_size = 2.0  # A.3 default for outdoor convergence basin

        self._well_posed_polish = well_posed_polish
        self._well_posed_kappa_threshold = well_posed_kappa_threshold
        self._well_posed_max_iters = well_posed_max_iters
        self._device = get_device(device)
        # AdaptiveFlatVoxelMap (C1):
        #   C1: world-frame merged voxel map with entropy-based adaptive resolution.
        #       Coarse voxels for flat/simple regions, fine for complex geometry.
        #       Combined splitting score S = H_geo + λ·H_int > τ_split.
        #   World-frame merging (Welford) gives accurate multi-frame covariances
        #   essential for reliable GICP in degenerate scenes (tunnels, corridors).
        # C3 retroactive correction would use LocalKeyframeVoxelMap (flat_voxel_map) if needed.
        # factor graph smoother (not yet integrated into the real-time odometry path).
        _map_frames = max_map_frames if max_map_frames is not None else max(max_map_points // 5000, 20)
        # C++ map: VoxelMap (fixed) or AdaptiveVoxelMap (C1). Same interface: insert_frame, build_target_arrays, query_sigma.
        # Falls back to Python FlatVoxelMap / AdaptiveFlatVoxelMap if C++ module not built.
        if _CPP_MAP_AVAILABLE:
            self._cpp_coarse_map = None  # set below if coarse enabled (non-adaptive only)
            if adaptive_voxelization:
                self._cpp_voxel_map = _cpp_map.AdaptiveVoxelMap(
                    voxel_size,
                    min_points_per_voxel,  # min_points_coarse
                    2,  # min_points_fine
                    _map_frames,
                    entropy_threshold,
                    0.1,  # lambda_intensity
                    min_eigenvalue_ratio,
                    max_condition_number,
                    adaptive_max_depth,
                )
            else:
                self._cpp_voxel_map = _cpp_map.VoxelMap(voxel_size, min_points_per_voxel)
            self.flat_map = None  # unused when C++ path active
        else:
            self._cpp_voxel_map = None
            self._cpp_coarse_map = None
            if adaptive_voxelization:
                raise RuntimeError(
                    "adaptive_voxelization=True requires the C++ extension (iv_gicp.cpp.iv_gicp_map). "
                    "Build with: python setup_cpp.py build_ext --inplace"
                )
            self.flat_map = FlatVoxelMap(voxel_size, min_points_per_voxel, _map_frames)
        self._map_frames = _map_frames
        self.iv_gicp = IVGICP(
            alpha=alpha,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            min_voxel_size=voxel_size,
            huber_delta=huber_delta,
            source_sigma=source_voxel_size / 2.0,
            device=device,
            use_fim_weight=use_fim_weight,
            use_degeneracy_aware_intensity_weight=use_degeneracy_aware_intensity_weight,
            degeneracy_kappa_threshold=degeneracy_kappa_threshold,
        )
        self._use_entropy_alpha = use_entropy_alpha
        self._entropy_scale_c = entropy_scale_c

        # Multi-scale: coarse map + ICP for better convergence basin
        self._coarse_map: Optional[FlatVoxelMap] = None
        self._coarse_iv_gicp: Optional[IVGICP] = None
        self._coarse_target_voxels_4d: Optional[List[Voxel4D]] = None
        self._coarse_target_means_3d: Optional[np.ndarray] = None
        self._coarse_target_tree: Optional[FastKDTree] = None
        self.coarse_voxel_size = coarse_voxel_size
        _coarse_max_corr = coarse_max_corr if coarse_max_corr > 0 else 3.0 * coarse_voxel_size
        self._coarse_max_corr = _coarse_max_corr
        # Coarse map: when adaptive is off, or when use_coarse_for_convergence (outdoor) is on.
        # A.3: outdoor/wide-basin — coarse→fine improves convergence; can be used with adaptive.
        if coarse_voxel_size > 0 and (not adaptive_voxelization or use_coarse_for_convergence):
            _coarse_map_frames = max(3, _map_frames // 3)
            if _CPP_MAP_AVAILABLE:
                self._cpp_coarse_map = _cpp_map.VoxelMap(coarse_voxel_size, 2)
                self._coarse_map = None
            else:
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
        # Schur complement marginalization: information from frames leaving the window
        # is propagated as a prior on the new oldest frame (no information loss).
        self._marginal_prior_omega: Optional[np.ndarray] = None
        self._marginal_prior_pose: Optional[np.ndarray] = None

        self.min_range = min_range
        self.max_range = max_range
        self.map_radius = map_radius  # spatial eviction radius; None = age-based
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
        self._use_conditional_intensity = use_conditional_intensity
        self._intensity_gradient_threshold = intensity_gradient_threshold
        self._auto_alpha_from_intensity = auto_alpha_from_intensity
        self._auto_alpha = auto_alpha
        self._kappa_threshold = kappa_threshold
        self._kappa_scale = kappa_scale
        self._alpha_floor_ratio = alpha_floor_ratio
        self._alpha_max = alpha  # when auto_alpha, effective_alpha in [0, _alpha_max]

        self.trajectory = Trajectory()
        self.current_pose = np.eye(4)
        self.map_voxels: Optional[List[tuple]] = None  # legacy (for C3 propagation)
        self.accumulated_points: Optional[np.ndarray] = None  # legacy (for ablation)

        # Cached registration targets (rebuilt from flat map)
        self._target_voxels_4d: Optional[List[Voxel4D]] = None
        self._target_means_3d: Optional[np.ndarray] = None
        self._target_tree: Optional[FastKDTree] = None
        self._cpp_target_arrays: Optional[dict] = None  # raw arrays for C++ path
        self._frame_count: int = 0
        # Rebuild KDTree + Voxel4D only every _kdtree_interval frames.
        # Between rebuilds: use cached tree (slightly stale but fast).
        self._kdtree_interval = kdtree_interval
        # auto_alpha: rebuild only when alpha changed enough or every N frames (speed).
        self._last_auto_alpha: Optional[float] = None
        self._last_alpha_rebuild_frame: int = -1
        self._auto_alpha_rebuild_interval: int = 2
        self._auto_alpha_change_threshold: float = 0.05

    def _prefilter(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
    ) -> tuple:
        """
        Range filter + voxel downsampling + intensity normalization.

        C++ fast path (when source_voxel_size > 0 and no range correction):
          Uses iv_gicp_map.downsample_and_filter — O(N) hash, single pass.
          ~20ms → ~1ms for 123k-point KITTI scans.

        Python fallback: lexsort + reduceat (O(N log N)).
        """
        pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        ints = np.ascontiguousarray(intensities, dtype=np.float64)

        # Range correction requires computing per-point ranges first → Python path
        if self.intensity_range_correction:
            ranges = np.linalg.norm(pts, axis=1)
            mask = (ranges > self.min_range) & (ranges < self.max_range)
            pts, ints = pts[mask], ints[mask]
            r0 = 5.0
            ints = ints * (ranges[mask] / r0) ** 2
            if self.source_voxel_size > 0 and len(pts) > 0:
                pts, ints = voxel_downsample(pts, ints, self.source_voxel_size)
            if len(ints) > 0:
                p99 = float(np.percentile(ints, 99))
                if p99 > 1.0:
                    ints = ints / p99
            return pts, ints

        # C++ fast path: O(N) range filter + hash downsample + intensity normalize
        if _CPP_MAP_AVAILABLE and self.source_voxel_size > 0:
            pts, ints = _cpp_map.downsample_and_filter(
                pts, ints, self.source_voxel_size, self.min_range, self.max_range
            )
            return pts, ints

        # Python fallback
        ranges = np.linalg.norm(pts, axis=1)
        mask = (ranges > self.min_range) & (ranges < self.max_range)
        pts, ints = pts[mask], ints[mask]
        if self.source_voxel_size > 0 and len(pts) > 0:
            pts, ints = voxel_downsample(pts, ints, self.source_voxel_size)
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
        Convert map to Voxel4D for direct registration.

        C++ path (fast): iv_gicp_map.VoxelMap.build_target_arrays() builds
        all arrays in one C++ pass (covariances + precision matrices + intensity
        gradients via nanoflann KNN). No Python listcomp or np.array() gather.

        Python fallback: FlatVoxelMap/AdaptiveFlatVoxelMap leaves + GPU batch.
        """
        alpha = self.iv_gicp.alpha

        if self._cpp_voxel_map is not None:
            # ── C++ fast path ───────────────────────────────────────────────
            # build_target_arrays returns contiguous numpy arrays AND caches the
            # nanoflann KDTree internally for query_sigma (no Python KDTree needed).
            arrays = self._cpp_voxel_map.build_target_arrays(
                alpha,
                self.source_voxel_size / 2.0,
                self.iv_gicp.n_grad_nbrs,
                2.0,  # count_reg_scale (matches gpu_backend default)
                self._use_entropy_alpha,
                self._entropy_scale_c,
            )
            V = len(arrays["means_4d"])
            if V == 0:
                return
            # Cache raw arrays — skip Voxel4D creation entirely
            self._cpp_target_arrays = arrays
            self._target_voxels_4d = None  # unused in C++ path
            self._target_means_3d = arrays["means_3d"]
            # FastKDTree only needed for coarse adaptive sigma fallback.
            # Primary sigma update uses _cpp_voxel_map.query_sigma (no Python overhead).
            self._target_tree = None  # will be built lazily if needed
            return

        # ── Python / GPU fallback ────────────────────────────────────────────
        leaf_stats = []
        for leaf in self.flat_map.leaves:
            s = leaf.stats
            if s is None or s.n_points < 2:
                continue
            leaf_stats.append((s, max(leaf.half_size * 2, 1e-3)))

        if not leaf_stats:
            return

        covs_arr = np.array([s.cov for s, _ in leaf_stats])
        var_i_arr = np.array([s.var_intensity for s, _ in leaf_stats])
        vsizes_arr = np.array([vs for _, vs in leaf_stats])
        n_counts = np.array([s.n_points for s, _ in leaf_stats])

        entropy_scale = None
        if self._use_entropy_alpha:
            _, log_det = np.linalg.slogdet(covs_arr + 1e-10 * np.eye(3))
            h_geo = 0.5 * log_det
            med = np.median(h_geo)
            entropy_scale = 1.0 + self._entropy_scale_c * (h_geo - med)
            entropy_scale = np.clip(entropy_scale, 0.4, 2.5).astype(np.float64)

        precisions = batch_precision_matrices(
            covs_arr,
            var_i_arr,
            vsizes_arr,
            alpha,
            source_sigma=self.source_voxel_size / 2.0,
            n_counts=n_counts,
            entropy_scale=entropy_scale,
            device=self._device,
        )

        n = len(leaf_stats)
        means_arr_4d = np.column_stack(
            [
                np.array([s.mean for s, _ in leaf_stats]),
                alpha * np.array([s.mean_intensity for s, _ in leaf_stats]),
            ]
        )
        means_3d = [s.mean for s, _ in leaf_stats]
        voxels_4d = [Voxel4D(mean=means_arr_4d[i], precision=precisions[i]) for i in range(n)]

        if not means_3d:
            return

        means_arr = np.array(means_3d)
        tree = FastKDTree(means_arr)
        self.iv_gicp._compute_intensity_gradients(means_arr, voxels_4d, tree)

        self._target_voxels_4d = voxels_4d
        self._target_means_3d = means_arr
        self._target_tree = tree

    def _build_coarse_voxels(self) -> None:
        """Build coarse-level Voxel4D targets from coarse FlatVoxelMap."""
        if self._cpp_coarse_map is not None:
            arrays = self._cpp_coarse_map.build_target_arrays(
                0.0,
                self.source_voxel_size / 2.0,
                self.iv_gicp.n_grad_nbrs,
                2.0,
            )
            means_4d = arrays["means_4d"]
            prec = arrays["prec"]
            grads = arrays["grads"]
            means_3d = arrays["means_3d"]
            V = len(means_4d)
            if V == 0:
                return
            voxels_4d = [Voxel4D(mean=means_4d[i], precision=prec[i], intensity_gradient=grads[i]) for i in range(V)]
            self._coarse_target_voxels_4d = voxels_4d
            self._coarse_target_means_3d = means_3d
            self._coarse_target_tree = FastKDTree(means_3d)
            return

        leaf_stats = []
        for leaf in self._coarse_map.leaves:
            s = leaf.stats
            if s is None or s.n_points < 2:
                continue
            leaf_stats.append((s, max(leaf.half_size * 2, 1e-3)))

        if not leaf_stats:
            return

        covs_arr = np.array([s.cov for s, _ in leaf_stats])
        var_i_arr = np.array([s.var_intensity for s, _ in leaf_stats])
        vsizes_arr = np.array([vs for _, vs in leaf_stats])
        n_counts = np.array([s.n_points for s, _ in leaf_stats])

        precisions = batch_precision_matrices(
            covs_arr,
            var_i_arr,
            vsizes_arr,
            0.0,
            source_sigma=self.source_voxel_size / 2.0,
            n_counts=n_counts,
            device=self._device,
        )

        n = len(leaf_stats)
        means_arr_4d = np.column_stack(
            [
                np.array([s.mean for s, _ in leaf_stats]),
                np.zeros(n),
            ]
        )
        means_3d = [s.mean for s, _ in leaf_stats]
        voxels_4d = [Voxel4D(mean=means_arr_4d[i], precision=precisions[i]) for i in range(n)]

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

        Uses flat_map or C++ VoxelMap; leaves are converted to Voxel4D and
        passed to IV-GICP, preserving variable-resolution covariances and intensity.

        Args:
            points:      (N, 3+) array with at least [x, y, z] columns
            intensities: (N,) intensity values; if None, extracted from points[:,3]
            init_pose:   override for initial pose estimate (default: constant-velocity)
            timestamp:   scan timestamp (optional)
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if intensities is None:
            intensities = points[:, 3] if points.shape[1] >= 4 else np.zeros(len(points))

        # Preprocessing: range filter + voxel downsampling
        points, intensities = self._prefilter(points, intensities)
        if len(points) < 10:
            pose = self.current_pose.copy()
            self.trajectory.poses.append(pose)
            self.trajectory.timestamps.append(timestamp or 0.0)
            return OdometryResult(pose=pose, timestamp=timestamp)

        # First frame: initialize map only (no registration target yet)
        if self._frame_count == 0:
            if self._cpp_voxel_map is not None:
                self._cpp_voxel_map.insert_frame(points[:, :3], intensities, 0)
            else:
                self.flat_map.insert_frame(points[:, :3], intensities, frame_id=0)
            self._build_voxels_from_adaptive_map()
            _cpp_cm = getattr(self, "_cpp_coarse_map", None)
            if _cpp_cm is not None:
                _cpp_cm.insert_frame(points[:, :3], intensities, 0)
                self._build_coarse_voxels()
            elif self._coarse_map is not None:
                self._coarse_map.insert_frame(points[:, :3], intensities, frame_id=0)
                self._build_coarse_voxels()
            self.map_voxels = []  # mark as initialized
            # C2 data-driven scale: alpha = 1/I_scale so residual ~ unit; uniform I → large alpha but omega_I small → cost small.
            if self._auto_alpha_from_intensity and self.iv_gicp.alpha > 0 and len(intensities) > 10:
                mad = np.median(np.abs(intensities - np.median(intensities)))
                i_scale = float(max(mad * 1.4826, 1e-9))
                self.iv_gicp.alpha = float(np.clip(1.0 / i_scale, 1e-6, 1e4))
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
        if (
            self._coarse_map is not None
            and self._coarse_target_tree is not None
            and self._coarse_target_voxels_4d is not None
        ):
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

        # C2 auto_alpha (Theorem 1): κ high → degenerate → use intensity; κ low → geometry-only
        # Rebuild target arrays only when alpha changed significantly or every N frames (speed).
        if self._auto_alpha and self._alpha_max > 0 and self._cpp_voxel_map is not None:
            kappa = self._cpp_voxel_map.get_max_condition_number()
            x = (kappa - self._kappa_threshold) / max(self._kappa_scale, 1e-6)
            sigmoid = 1.0 / (1.0 + np.exp(-float(x)))
            effective_alpha = self._alpha_max * sigmoid
            if kappa > self._kappa_threshold:
                floor = self._alpha_max * self._alpha_floor_ratio
                effective_alpha = max(effective_alpha, floor)
            self.iv_gicp.alpha = float(np.clip(effective_alpha, 0.0, self._alpha_max))
            do_rebuild = False
            if self._last_auto_alpha is None:
                do_rebuild = True
            elif abs(effective_alpha - self._last_auto_alpha) > self._auto_alpha_change_threshold:
                do_rebuild = True
            elif (effective_alpha == 0.0) != (self._last_auto_alpha == 0.0):
                do_rebuild = True
            elif self._frame_count - self._last_alpha_rebuild_frame >= self._auto_alpha_rebuild_interval:
                do_rebuild = True
            if do_rebuild:
                self._build_voxels_from_adaptive_map()
                self._last_auto_alpha = effective_alpha
                self._last_alpha_rebuild_frame = self._frame_count

        # Register source → adaptive voxel map (direct path)
        reg_info = {"n_correspondences": 0, "converged": False}
        # C2 conditional: use intensity only when map has informative gradients.
        # In uniform-intensity scenes (e.g. metro tunnel), intensity adds noise → use geometry-only.
        effective_alpha = self.iv_gicp.alpha
        if self._use_conditional_intensity and self.iv_gicp.alpha > 0 and not self._auto_alpha:
            mean_grad = 0.0
            if self._cpp_target_arrays is not None and "grads" in self._cpp_target_arrays:
                grads = np.asarray(self._cpp_target_arrays["grads"])
                if len(grads) > 0:
                    mean_grad = float(np.mean(np.linalg.norm(grads, axis=1)))
            elif self._target_voxels_4d is not None:
                grads = np.array([v.intensity_gradient for v in self._target_voxels_4d])
                if len(grads) > 0:
                    mean_grad = float(np.mean(np.linalg.norm(grads, axis=1)))
            if mean_grad < self._intensity_gradient_threshold:
                effective_alpha = 0.0
        _alpha_restore = None
        if effective_alpha != self.iv_gicp.alpha:
            _alpha_restore = self.iv_gicp.alpha
            self.iv_gicp.alpha = effective_alpha
        _t_reg = time.perf_counter()
        if self._cpp_target_arrays is not None:
            # C++ fast path: raw arrays bypass Voxel4D extraction overhead
            T, reg_info = self.iv_gicp.register_with_arrays(
                points[:, :3],
                intensities,
                self._cpp_target_arrays,
                init_pose=init_pose,
                max_corr_dist=adaptive_corr,
            )
        elif self._target_voxels_4d is not None and self._target_tree is not None:
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
                    source_points,
                    target_points,
                    source_intensities=intensities,
                    target_intensities=target_intensities,
                    init_pose=init_pose,
                )
                reg_info = {"n_correspondences": 999, "converged": True}
        if _alpha_restore is not None:
            self.iv_gicp.alpha = _alpha_restore
        reg_ms = (time.perf_counter() - _t_reg) * 1000

        # A.1 Well-posed polish: when geometry is well-conditioned (κ < threshold), refine with geometry-only 1–2 GN steps
        if (
            self._well_posed_polish
            and reg_info.get("converged")
            and reg_info.get("hessian") is not None
        ):
            H = np.asarray(reg_info["hessian"])
            kappa = np.linalg.cond(H)
            if kappa < self._well_posed_kappa_threshold:
                old_alpha = self.iv_gicp.alpha
                old_max_iter = self.iv_gicp.max_iter
                self.iv_gicp.alpha = 0.0
                self.iv_gicp.max_iter = self._well_posed_max_iters
                try:
                    if self._cpp_target_arrays is not None:
                        T, _ = self.iv_gicp.register_with_arrays(
                            points[:, :3],
                            intensities,
                            self._cpp_target_arrays,
                            init_pose=T,
                            max_corr_dist=adaptive_corr,
                        )
                    elif self._target_voxels_4d is not None and self._target_tree is not None:
                        T, _ = self.iv_gicp.register_with_voxel_map(
                            np.column_stack([points[:, :3], intensities]),
                            intensities,
                            self._target_voxels_4d,
                            self._target_means_3d,
                            self._target_tree,
                            init_pose=T,
                            max_corr_dist=adaptive_corr,
                        )
                finally:
                    self.iv_gicp.alpha = old_alpha
                    self.iv_gicp.max_iter = old_max_iter

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
                T = self._window_smooth(T, T_rel_cur, H_icp, self._frame_count, n_valid=n_corr)

        # Update adaptive sigma from accepted registration quality.
        # sigma = median correspondence distance between accepted-pose source and target.
        # This directly measures map-to-scan alignment quality for the next frame's threshold.
        # Using correspondence distances (not velocity deviation) is more stable because
        # it measures actual map quality independent of motion estimation errors.
        if len(points) > 5:
            pts_world = (T[:3, :3] @ points[:, :3].T).T + T[:3, 3]
            if self._cpp_voxel_map is not None:
                # C++ path: query cached nanoflann tree, compute median in C++
                sigma_new = self._cpp_voxel_map.query_sigma(pts_world, adaptive_corr, self._min_motion_th)
                self._adaptive_sigma = sigma_new
            elif self._target_tree is not None:
                dists, _ = self._target_tree.query(pts_world, k=1)
                inlier_mask = dists < adaptive_corr
                if np.sum(inlier_mask) > 5:
                    sigma_new = max(float(np.median(dists[inlier_mask])), self._min_motion_th)
                    self._adaptive_sigma = sigma_new

        self.current_pose = T
        self.trajectory.poses.append(self.current_pose.copy())
        self.trajectory.timestamps.append(timestamp or self.trajectory.timestamps[-1] + 1.0)

        # Only update the map when registration quality was sufficient.
        # Skipping map update on poor frames prevents corrupted points from
        # accumulating and causing cascade failures on subsequent frames.
        _t_map = time.perf_counter()
        if not poor_registration:
            self._update_map(points, intensities, T, self._frame_count)
        map_ms = (time.perf_counter() - _t_map) * 1000
        self._frame_count += 1

        return OdometryResult(
            pose=T,
            timestamp=timestamp,
            num_correspondences=n_corr,
            reg_ms=reg_ms,
            map_ms=map_ms,
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

        if self._cpp_voxel_map is not None:
            # ── C++ fast path ────────────────────────────────────────────────
            self._cpp_voxel_map.insert_frame(pts_world, intensities, frame_id)
            if self._frame_count % 10 == 0:
                if self.map_radius is not None:
                    # Spatial eviction (KISS-ICP style): remove voxels farther than
                    # map_radius from the current robot. Best for slow or looping
                    # trajectories (tunnels, corridors) where age-based eviction
                    # discards nearby voxels while keeping stale distant ones.
                    cx, cy, cz = T[0, 3], T[1, 3], T[2, 3]
                    self._cpp_voxel_map.evict_far_from(cx, cy, cz, self.map_radius)
                else:
                    # Age-based eviction: keep the most recent max_frames keyframes.
                    evict_before = frame_id - self._map_frames
                    if evict_before > 0:
                        self._cpp_voxel_map.evict_before(evict_before)
        else:
            # ── Python fallback ──────────────────────────────────────────────
            self.flat_map.insert_frame(pts_world, intensities, frame_id)
            if self._frame_count % 10 == 0:
                evict_before = self._frame_count - self.flat_map.max_frames
                if evict_before > 0:
                    self.flat_map.evict_before(evict_before)

        if self._frame_count % self._kdtree_interval == 0 or self._target_tree is None:
            self._build_voxels_from_adaptive_map()

        # Coarse map
        _cpp_cm = getattr(self, "_cpp_coarse_map", None)
        if _cpp_cm is not None:
            _cpp_cm.insert_frame(pts_world, intensities, self._frame_count)
            if self._frame_count % 10 == 0:
                if self.map_radius is not None:
                    cx, cy, cz = T[0, 3], T[1, 3], T[2, 3]
                    _cpp_cm.evict_far_from(cx, cy, cz, self.map_radius)
                else:
                    evict_before_cc = self._frame_count - self._map_frames
                    if evict_before_cc > 0:
                        _cpp_cm.evict_before(evict_before_cc)
            if self._frame_count % self._kdtree_interval == 0 or self._coarse_target_tree is None:
                self._build_coarse_voxels()
        elif self._coarse_map is not None:
            self._coarse_map.insert_frame(pts_world, intensities, self._frame_count)
            if self._frame_count % 10 == 0:
                evict_before_c = self._frame_count - self._coarse_map.max_frames
                if evict_before_c > 0:
                    self._coarse_map.evict_before(evict_before_c)
            if self._frame_count % self._kdtree_interval == 0 or self._coarse_target_tree is None:
                self._build_coarse_voxels()

    def _window_smooth(
        self,
        T_new: np.ndarray,
        T_rel: np.ndarray,
        H: np.ndarray,
        frame_id: int,
        n_valid: int = 0,
    ) -> np.ndarray:
        """
        Intensity-augmented FORM-style fixed-lag window smoothing.

        Improvements over FORM (Potokar et al. 2025, arXiv 2510.09966):

        [1] Intensity-augmented Hessian (C2 × FORM synergy):
            H includes photometric Fisher information (geo + α·photo).
            Even when all frames share the same geometric degeneracy direction,
            intensity gradients differ → window correction still possible.
            FORM with geometry-only Hessian cannot exploit this.

        [2] Quality-weighted information matrices:
            Each frame's H scaled by n_valid_i / n_valid_max.
            Low-correspondence frames (unreliable measurements) contribute
            proportionally less to the joint optimization.

        [3] Adaptive window bypass:
            condition number κ = λ_max / λ_min of current H.
            κ < 100: all DOFs well-constrained → window adds no information → skip.
            Eliminates overhead for non-degenerate environments entirely.

        [4] Schur complement marginalization:
            When oldest frame leaves window, its information is propagated as a
            prior on the new oldest frame via Schur complement (not discarded).
            Standard fixed-lag smoother behavior: no information loss.
            omega_marginal = H_01 - H_01^T (H_00)^{-1} H_01
        """
        self._window_buffer.append((T_new.copy(), T_rel.copy(), H.copy(), frame_id, n_valid))
        K = len(self._window_buffer)
        if K < 2:
            return T_new

        buf = list(self._window_buffer)

        # [3] Adaptive bypass: skip if current measurement is well-constrained.
        H_cur = (H + H.T) * 0.5
        eigs_cur = np.linalg.eigvalsh(H_cur)
        kappa = float(eigs_cur[-1]) / max(float(eigs_cur[0]), 1e-4)
        if kappa < 100.0:
            return T_new

        # [2] Quality normalization factor.
        n_valids = [e[4] for e in buf]
        n_max = max(n_valids) if max(n_valids) > 0 else 1

        optimizer = PoseGraphOptimizer(
            lag_size=K,
            max_iterations=3,
            convergence_threshold=1e-6,
            prior_weight=1e4,
        )

        # [4] Apply marginalized prior from previous window step (if available).
        prior_omega = self._marginal_prior_omega if self._marginal_prior_omega is not None else np.eye(6) * 1e4
        optimizer.set_prior(buf[0][0], omega=prior_omega)

        for T_abs, _, _, _, _ in buf:
            optimizer.add_pose(T_abs)

        for i in range(1, K):
            T_rel_i = buf[i][1]
            H_i = buf[i][2]
            # [2] Scale H by correspondence quality ratio.
            q_i = buf[i][4] / n_max if n_max > 0 else 1.0
            H_sym = (H_i + H_i.T) * 0.5 * q_i
            min_eig = float(np.linalg.eigvalsh(H_sym)[0])
            if min_eig < 1e-4:
                H_sym = H_sym + (1e-4 - min_eig) * np.eye(6)
            optimizer.add_odometry_factor(i - 1, i, T_rel_i, omega=H_sym)

        opt_poses = optimizer.optimize()
        if opt_poses is None or len(opt_poses) < K:
            return T_new

        # Retroactively correct stored trajectory poses.
        n_traj = len(self.trajectory.poses)
        for k in range(K - 1):
            traj_idx = n_traj - (K - 1) + k
            if 0 <= traj_idx < n_traj:
                self.trajectory.poses[traj_idx] = opt_poses[k]

        # [4] Schur complement marginalization: when buffer is full, compute marginal
        # prior on pose 1 before pose 0 is evicted. Approx: J_i ≈ -I, J_j ≈ I (near
        # convergence), so H_00 block = prior_omega + H_f01, H_01 block = -H_f01.
        if K >= self._window_size:
            q_1 = buf[1][4] / n_max if n_max > 0 else 1.0
            H_f01 = (buf[1][2] + buf[1][2].T) * 0.5 * q_1
            min_e = float(np.linalg.eigvalsh(H_f01)[0])
            if min_e < 1e-4:
                H_f01 = H_f01 + (1e-4 - min_e) * np.eye(6)
            H_00_total = prior_omega + H_f01
            H_off = -H_f01
            try:
                H_00_inv = np.linalg.inv(H_00_total)
                omega_new = H_f01 - H_off.T @ H_00_inv @ H_off
                eigs_m = np.linalg.eigvalsh(omega_new)
                if eigs_m[0] < 1e-8:
                    omega_new += (1e-8 - eigs_m[0]) * np.eye(6)
                self._marginal_prior_omega = omega_new
                self._marginal_prior_pose = opt_poses[1].copy()
            except np.linalg.LinAlgError:
                pass  # keep old prior on failure

        # Update buffer with corrected poses for next frame's T_rel computation.
        new_buf = [(opt_poses[k], buf[k][1], buf[k][2], buf[k][3], buf[k][4]) for k in range(K)]
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
        covs = [v[1] for v in self.map_voxels]
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
