"""
IV-GICP Full Pipeline

PointCloud → [range filter] → [C++ voxel PCA plane/edge (C4)] →
constant-velocity prediction → C++ VoxelMap → C++ IV-GICP registration (bindings only).

Requires built extensions: ``python setup_cpp.py build_ext --inplace``.

Key improvements over draft:
  - Constant-velocity motion prediction for ICP initial pose (like KISS-ICP)
  - Range filtering to remove close-range noise and far-range outliers
  - Sliding-window map: limits accumulated points to prevent O(N·logN) rebuild growth
"""

import time

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from .iv_gicp import IVGICP
from .se3_utils import se3_inverse, se3_compose, se3_log
from .gpu_backend import get_device

from .cpp import iv_gicp_map as _cpp_map
from .cpp import iv_gicp_core as _cpp_core

_VOXEL_DOWNSPAMPLE_CPP = getattr(_cpp_core, "voxel_downsample", None)


def voxel_downsample(
    points: np.ndarray,
    intensities: np.ndarray,
    voxel_size: float,
) -> tuple:
    """Voxel centroid downsampling via C++ iv_gicp_core."""
    pts = np.asarray(points[:, :3], dtype=np.float64, order="C")
    ints = np.asarray(intensities, dtype=np.float64, order="C")
    out_pts, out_ints = _VOXEL_DOWNSPAMPLE_CPP(pts, ints, float(voxel_size))
    return np.asarray(out_pts), np.asarray(out_ints)


@dataclass
class OdometryResult:
    """Single frame odometry result."""

    pose: np.ndarray  # 4×4 absolute pose (world ← sensor)
    timestamp: Optional[float] = None
    num_correspondences: int = 0
    # Per-phase timing (ms), 0.0 if not measured
    reg_ms: float = 0.0  # ICP registration time
    map_ms: float = 0.0  # map insert + optional KDTree rebuild time
    odometry_mode: str = "iv"
    kappa: float = 0.0        # GN Hessian condition number (κ = λ_max/λ_min); large → degenerate geometry
    mscs_ratio: float = 1.0   # n_mscs_used / n_correspondences (1.0 = all used; MSCS disabled)


@dataclass
class Trajectory:
    """Trajectory and map state."""

    poses: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class IVGICPPipeline:
    """
    Full IV-GICP odometry pipeline.

    Processes LiDAR scans sequentially and maintains a growing voxel map.
    Uses constant-velocity motion model for initial pose prediction.
    """

    def __init__(
        self,
        voxel_size: float = 1.0,
        intensity_range_correction: bool = False,  # I_cal = I × r² / r0² (range-invariant reflectivity)
        min_points_per_voxel: int = 3,
        # IV-GICP registration
        alpha: float = 0.1,  # corrected from 0.01: ~range/max_I
        max_correspondence_distance: float = 2.0,
        max_iterations: int = 12,
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
        # C4 source: C++ voxel_downsample_plane_edge (PCA plane/edge + sphericity; FIM triggers below).
        source_plane_planarity_thresh: float = 0.12,
        source_plane_linearity_thresh: float = 0.12,
        source_plane_edge_sphere_thresh: float = 0.5,
        source_plane_n_min: int = 200,
        # C4 sparse confident: drop undersampled voxels; keep top plane/edge scores (C++).
        # source_drop_small_voxels=False: dropping sparse voxels is too aggressive and hurts ATE.
        source_drop_small_voxels: bool = False,
        source_max_output_features: int = 0,  # 0 = no cap after scoring
        source_min_feature_score: float = 0.0,  # min P (plane) or L (edge); 0 = off
        source_fim_edge_lambda_min: float = 0.0,  # >0: force edge pool if min eig(H) < this
        source_fim_min_corr: int = 0,  # >0: force edge pool if prev n_corr < this
        # C2: intensity scale from data so C2 doesn't hurt in uniform-intensity scenes (no fallback to GICP)
        use_conditional_intensity: bool = False,  # if True: use geometry-only when mean(|∇I|) < threshold
        intensity_gradient_threshold: float = 1e-6,
        # auto_alpha_from_intensity=False: True causes alpha≈6.7 on KITTI [0,1] intensities (MAD≈0.15 → alpha=1/0.15≈6.7), destroying ATE.
        auto_alpha_from_intensity: bool = False,
        # auto_alpha=False: kappa-based sigmoid adaptation hurts KITTI seq00 (0.313→0.617m). Use fixed per-domain alpha.
        auto_alpha: bool = False,
        kappa_threshold: float = 10.0,  # κ above this → increase alpha (tunnel/degenerate); lowered from 15 for earlier intensity use
        kappa_scale: float = 7.0,  # sigmoid steepness (gentler so alpha rises in mid-κ)
        alpha_floor_ratio: float = 0.2,  # when κ > threshold, effective_alpha >= alpha_max * this (avoid too-weak intensity in degenerate)
        use_fim_weight: bool = False,
        use_entropy_alpha: bool = False,
        entropy_scale_c: float = 0.5,  # scale = 1 + c * (h_geo - median(h_geo)), clipped [0.4, 2.5]; was 0.3 for stronger C3
        # A.1 Well-posed polish: when κ_geo < threshold, refine with geometry-only 1–2 GN steps
        well_posed_polish: bool = False,
        well_posed_kappa_threshold: float = 100.0,
        well_posed_max_iters: int = 2,
        device: str = "cpu",
        # GN cap: C++ subsamples source correspondences. 0 = use all (recommended for accuracy).
        # 2048 is a speed preset; use 0 for paper-grade ATE.
        max_source_points: int = 0,
        # OpenMP thread count (0 = system default / OMP_NUM_THREADS env var)
        omp_num_threads: int = 0,
        # [I2] Geman-McClure kernel scale c; w=c²/(c²+r²) on Mahalanobis residual r; 0=disabled
        gm_scale: float = 0.0,
        # [I3] FIM hard gate: drop correspondences with FIM_contribution < ratio×mean_FIM; 0=disabled
        fim_gate_ratio: float = 0.0,
        # [C4b] Trimmed GICP: in GN registration, drop top trim_ratio fraction of correspondences
        # ranked by Mahalanobis residual (dynamic objects have large residual → trimmed)
        # 0.0 = disabled; 0.2 = trim top 20%
        icp_trim_ratio: float = 0.0,
        # [C4c] Ghost raycast eviction: free-space consistency check via DDA ray marching.
        # If a map voxel is contradicted by ≥ min_pass_through rays from the current scan,
        # it is evicted as a ghost (stale dynamic object distribution).
        # 0.0 = disabled; typical value 0.5 (evict if >50% of rays pass through it)
        ghost_raycast_threshold: float = 0.0,
        ghost_min_pass_through: int = 3,
        # [C4-TSG] Temporal Stability Gating: map voxels with n_frames < stable_frames
        # get their precision matrix blended toward isotropic (P2P-equivalent).
        # 0 = disabled; default 10 (see docs/c4_feature_selective_gicp.md).
        stable_frames: int = 10,
        # EMA forgetting window for map voxels. KISS uses 20; IV-GICP default 100.
        # Smaller = ghost means fade faster. 0 = unlimited Welford.
        map_max_n: int = 100,
        # C++/map: cap target voxels passed to GN + KD-tree (ranked by per-voxel point count n).
        # 0 = no cap (paper default). >0 = LOAM-style lighter registration target (map insert unchanged).
        max_registration_target_voxels: int = 0,
        # C4 MSCS: Minimum Sufficient Correspondence Set
        # Stop accumulating correspondences once λ_min(H) ≥ λ_max(H)/mscs_kappa_max.
        # Derived from Theorem 1 well-posedness condition — no dataset-specific tuning.
        use_mscs: bool = False,
        mscs_kappa_max: float = 100.0,
        # Voxel covariance regularization: adds (count_reg_scale²/n)·I to each voxel covariance.
        # Lower → tighter initial covariance → better correspondence quality for new voxels.
        # Default 2.0 (conservative); 1.0 improves early-frame accuracy.
        count_reg_scale: float = 2.0,
        **kwargs: object,
    ):
        _ = kwargs  # absorb legacy keyword args from old configs/scripts
        # Merge config from YAML (if present) so callers can rely on config/ for defaults
        try:
            from .config_loader import get_pipeline_config
            _file_cfg = get_pipeline_config()
        except Exception:
            _file_cfg = {}
        _opts = dict(
            voxel_size=voxel_size,
            intensity_range_correction=intensity_range_correction,
            min_points_per_voxel=min_points_per_voxel,
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
            coarse_voxel_size=coarse_voxel_size,
            coarse_iterations=coarse_iterations,
            coarse_max_corr=coarse_max_corr,
            source_plane_planarity_thresh=source_plane_planarity_thresh,
            source_plane_linearity_thresh=source_plane_linearity_thresh,
            source_plane_edge_sphere_thresh=source_plane_edge_sphere_thresh,
            source_plane_n_min=source_plane_n_min,
            source_drop_small_voxels=source_drop_small_voxels,
            source_max_output_features=source_max_output_features,
            source_min_feature_score=source_min_feature_score,
            source_fim_edge_lambda_min=source_fim_edge_lambda_min,
            source_fim_min_corr=source_fim_min_corr,
            use_conditional_intensity=use_conditional_intensity,
            intensity_gradient_threshold=intensity_gradient_threshold,
            auto_alpha_from_intensity=auto_alpha_from_intensity,
            auto_alpha=auto_alpha,
            kappa_threshold=kappa_threshold,
            kappa_scale=kappa_scale,
            alpha_floor_ratio=alpha_floor_ratio,
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
            max_source_points=max_source_points,
            device=device,
            omp_num_threads=omp_num_threads,
            gm_scale=gm_scale,
            fim_gate_ratio=fim_gate_ratio,
            icp_trim_ratio=icp_trim_ratio,
            ghost_raycast_threshold=ghost_raycast_threshold,
            ghost_min_pass_through=ghost_min_pass_through,
            stable_frames=stable_frames,
            map_max_n=map_max_n,
            max_registration_target_voxels=max_registration_target_voxels,
            use_mscs=use_mscs,
            mscs_kappa_max=mscs_kappa_max,
            count_reg_scale=count_reg_scale,
        )
        # Explicit constructor args override config file
        _opts = {**_file_cfg, **_opts}
        voxel_size = _opts["voxel_size"]
        intensity_range_correction = _opts["intensity_range_correction"]
        min_points_per_voxel = _opts["min_points_per_voxel"]
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
        coarse_voxel_size = _opts["coarse_voxel_size"]
        coarse_iterations = _opts["coarse_iterations"]
        coarse_max_corr = _opts["coarse_max_corr"]
        source_plane_planarity_thresh = float(_opts.get("source_plane_planarity_thresh", 0.12))
        source_plane_linearity_thresh = float(_opts.get("source_plane_linearity_thresh", 0.12))
        source_plane_edge_sphere_thresh = float(_opts.get("source_plane_edge_sphere_thresh", 0.5))
        source_plane_n_min = int(_opts.get("source_plane_n_min", 200))
        source_drop_small_voxels = bool(_opts.get("source_drop_small_voxels", True))
        source_max_output_features = int(_opts.get("source_max_output_features", 2048))
        source_min_feature_score = float(_opts.get("source_min_feature_score", 0.0))
        source_fim_edge_lambda_min = float(_opts.get("source_fim_edge_lambda_min", 0.0))
        source_fim_min_corr = int(_opts.get("source_fim_min_corr", 0))
        use_conditional_intensity = _opts["use_conditional_intensity"]
        intensity_gradient_threshold = _opts["intensity_gradient_threshold"]
        auto_alpha_from_intensity = _opts["auto_alpha_from_intensity"]
        auto_alpha = _opts["auto_alpha"]
        kappa_threshold = _opts["kappa_threshold"]
        kappa_scale = _opts["kappa_scale"]
        alpha_floor_ratio = _opts["alpha_floor_ratio"]
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
        max_source_points = int(_opts.get("max_source_points", 0))
        omp_num_threads = int(_opts.get("omp_num_threads", 0))
        device = _opts["device"]
        if use_coarse_for_convergence and coarse_voxel_size <= 0:
            coarse_voxel_size = 2.0  # A.3 default for outdoor convergence basin

        self._well_posed_polish = well_posed_polish
        self._well_posed_kappa_threshold = well_posed_kappa_threshold
        self._well_posed_max_iters = well_posed_max_iters
        self._device = get_device(device)
        # Default to 4 threads when not specified: avoids OMP overhead with 64+ threads
        # on machines where sparse source (~2048 pts) makes per-thread work too small.
        _omp_threads = omp_num_threads if omp_num_threads > 0 else 4
        try:
            _cpp_map.set_num_threads(_omp_threads)
        except Exception:
            pass
        try:
            _cpp_core.set_num_threads(_omp_threads)
        except Exception:
            import os
            os.environ["OMP_NUM_THREADS"] = str(_omp_threads)
        _map_frames = max_map_frames if max_map_frames is not None else max(max_map_points // 5000, 20)
        self._cpp_coarse_map = None
        self._cpp_coarse_target_arrays: Optional[dict] = None
        try:
            self._cpp_voxel_map = _cpp_map.VoxelMap(
                voxel_size, min_points_per_voxel, _opts.get("map_max_n", 100)
            )
        except TypeError:
            # Committed .so: VoxelMap only takes 2 args
            self._cpp_voxel_map = _cpp_map.VoxelMap(voxel_size, min_points_per_voxel)
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
            max_source_points=max_source_points,
            gm_scale=_opts.get("gm_scale", 0.0),
            fim_gate_ratio=_opts.get("fim_gate_ratio", 0.0),
            icp_trim_ratio=_opts.get("icp_trim_ratio", icp_trim_ratio),
        )
        self._use_entropy_alpha = use_entropy_alpha
        self._entropy_scale_c = entropy_scale_c
        self._ghost_raycast_threshold = float(_opts.get("ghost_raycast_threshold", ghost_raycast_threshold))
        self._ghost_min_pass_through = int(_opts.get("ghost_min_pass_through", ghost_min_pass_through))
        self._stable_frames = int(_opts.get("stable_frames", stable_frames))
        self._max_registration_target_voxels = int(_opts.get("max_registration_target_voxels", 0))

        self._use_mscs = bool(_opts.get("use_mscs", False))
        self._mscs_kappa_max = float(_opts.get("mscs_kappa_max", 100.0))
        self._prev_v_min: np.ndarray = np.zeros(6, dtype=np.float64)  # warm-start for MSCS
        self._count_reg_scale = float(_opts.get("count_reg_scale", 2.0))

        self._coarse_iv_gicp: Optional[IVGICP] = None
        self.coarse_voxel_size = coarse_voxel_size
        _coarse_max_corr = coarse_max_corr if coarse_max_corr > 0 else 3.0 * coarse_voxel_size
        self._coarse_max_corr = _coarse_max_corr
        if coarse_voxel_size > 0:
            self._cpp_coarse_map = _cpp_map.VoxelMap(coarse_voxel_size, 2)
            self._coarse_iv_gicp = IVGICP(
                alpha=0.0,
                max_correspondence_distance=_coarse_max_corr,
                max_iterations=coarse_iterations,
                min_voxel_size=coarse_voxel_size,
                huber_delta=0.0,
                source_sigma=source_voxel_size / 2.0,
                device=device,
            )

        self._source_plane_planarity_thresh = source_plane_planarity_thresh
        self._source_plane_linearity_thresh = source_plane_linearity_thresh
        self._source_plane_edge_sphere_thresh = source_plane_edge_sphere_thresh
        self._source_plane_n_min = source_plane_n_min
        self._source_drop_small_voxels = source_drop_small_voxels
        self._source_max_output_features = source_max_output_features
        self._source_min_feature_score = source_min_feature_score
        self._source_fim_edge_lambda_min = source_fim_edge_lambda_min
        self._source_fim_min_corr = source_fim_min_corr
        self._prev_h_min_eig: Optional[float] = None
        self._prev_n_corr: Optional[int] = None

        self.min_range = min_range
        self.max_range = max_range
        self.map_radius = map_radius  # spatial eviction radius; None = age-based
        self.source_voxel_size = source_voxel_size
        self.max_map_points = max_map_points
        self.intensity_range_correction = intensity_range_correction

        # [I1] Adaptive threshold — KISS-ICP exact formula.
        # sigma = sqrt(SSE_model_deviation / N) = RMS of pose-prediction errors.
        # Tight when tracking is good (small deviations → tighter correspondences).
        # Wide when bad (large deviations → recovery mode, wider search).
        # Do NOT use correspondence-distance sigma: that creates a positive feedback
        # loop (drift → far correspondences → larger threshold → more drift).
        # [I1] KISS-ICP compatible adaptive sigma.
        # Warm-start: sigma = initial_threshold / 3 (exact KISS-ICP formula).
        # Lower bound = min_motion_th (0.1m default, same as KISS-ICP).
        _initial_sigma = initial_threshold / 3.0
        self._model_error_sse2 = _initial_sigma ** 2  # warm-start SSE
        self._model_n_samples = 1
        self._model_dev_sigma = _initial_sigma
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
        self._cpp_target_arrays: Optional[dict] = None  # raw arrays for C++ path
        self._registration_session = None  # C++ RegistrationSession: reuse target KDTree when target unchanged
        self._frame_count: int = 0
        # Rebuild KDTree + Voxel4D only every _kdtree_interval frames.
        # Between rebuilds: use cached tree (slightly stale but fast).
        self._kdtree_interval = kdtree_interval
        # auto_alpha: rebuild only when alpha changed enough or every N frames (speed).
        self._last_auto_alpha: Optional[float] = None
        self._last_alpha_rebuild_frame: int = -1
        self._auto_alpha_rebuild_interval: int = self._kdtree_interval  # sync with kdtree rebuild
        self._auto_alpha_change_threshold: float = 0.05

    def _prefilter(
        self,
        points: np.ndarray,
        intensities: np.ndarray,
    ) -> tuple:
        """
        Range filter + voxel downsampling + intensity normalization.

        C++ path (when source_voxel_size > 0 and no range correction):
          iv_gicp_map.voxel_downsample_plane_edge — C4 PCA plane/edge + sphericity filter.

        Python fallback (no C++ map module): centroid voxel grid only (no plane/edge split).
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

        # C++ path: C4 plane/edge + sphericity (voxel_downsample_plane_edge)
        if self.source_voxel_size > 0:
            force_edges = False
            if self._source_fim_edge_lambda_min > 0.0 and self._prev_h_min_eig is not None:
                if self._prev_h_min_eig < self._source_fim_edge_lambda_min:
                    force_edges = True
            if self._source_fim_min_corr > 0 and self._prev_n_corr is not None:
                if self._prev_n_corr < self._source_fim_min_corr:
                    force_edges = True
            pts_c4_in = pts
            ints_c4_in = ints
            if not hasattr(_cpp_map, "voxel_downsample_plane_edge"):
                pts, ints = _cpp_map.downsample_and_filter(
                    pts_c4_in, ints_c4_in, self.source_voxel_size, self.min_range, self.max_range
                )
                return pts, ints
            pts, ints = _cpp_map.voxel_downsample_plane_edge(
                pts,
                ints,
                self.source_voxel_size,
                self.min_range,
                self.max_range,
                self._source_plane_edge_sphere_thresh,
                4,
                self._source_plane_planarity_thresh,
                self._source_plane_linearity_thresh,
                int(self._source_plane_n_min),
                bool(force_edges),
                bool(self._source_drop_small_voxels),
                int(self._source_max_output_features),
                float(self._source_min_feature_score),
            )
            # Degenerate scan: strict C4 can return almost no points
            if pts.shape[0] < 10:
                pts, ints = _cpp_map.downsample_and_filter(
                    pts_c4_in, ints_c4_in, self.source_voxel_size, self.min_range, self.max_range
                )
            return pts, ints

        # source_voxel_size == 0: range filter + intensity normalize only
        ranges = np.linalg.norm(pts, axis=1)
        mask = (ranges > self.min_range) & (ranges < self.max_range)
        pts, ints = pts[mask], ints[mask]
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

    def _build_voxels_from_cpp_map(self) -> None:
        """Build target arrays from C++ VoxelMap.build_target_arrays."""
        alpha = self.iv_gicp.alpha
        if self._cpp_target_arrays is not None:
            self._cpp_target_arrays = None
            self._registration_session = None
        try:
            arrays = self._cpp_voxel_map.build_target_arrays(
                alpha,
                self.source_voxel_size / 2.0,
                self.iv_gicp.n_grad_nbrs,
                self._count_reg_scale,
                self._use_entropy_alpha,
                self._entropy_scale_c,
                self._stable_frames,
                self._max_registration_target_voxels,
            )
        except TypeError:
            # Committed .so: fewer args
            arrays = self._cpp_voxel_map.build_target_arrays(
                alpha,
                self.source_voxel_size / 2.0,
                self.iv_gicp.n_grad_nbrs,
                self._count_reg_scale,
                self._use_entropy_alpha,
                self._entropy_scale_c,
            )
        if len(arrays["means_4d"]) == 0:
            return
        self._cpp_target_arrays = arrays
        if getattr(_cpp_core, "RegistrationSession", None) is not None:
            self._registration_session = _cpp_core.RegistrationSession(
                np.ascontiguousarray(arrays["means_3d"], dtype=np.float64)
            )

    def _build_coarse_voxels(self) -> None:
        """Cache coarse target arrays for register_with_arrays."""
        if self._cpp_coarse_map is None or self._coarse_iv_gicp is None:
            return
        arrays = self._cpp_coarse_map.build_target_arrays(
            0.0,
            self.source_voxel_size / 2.0,
            self.iv_gicp.n_grad_nbrs,
            self._count_reg_scale,
            self._use_entropy_alpha,
            self._entropy_scale_c,
            0,
            self._max_registration_target_voxels,
        )
        if len(arrays["means_4d"]) == 0:
            self._cpp_coarse_target_arrays = None
            return
        self._cpp_coarse_target_arrays = arrays

    def process_frame(
        self,
        points: np.ndarray,
        intensities: Optional[np.ndarray] = None,
        init_pose: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> OdometryResult:
        """
        Process one LiDAR scan. Returns absolute pose and updates internal map.

        C++ VoxelMap + C++ registration (iv_gicp_core).

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

        # Map source: range-filtered raw points (dense, not C4-filtered).
        # Registration source (below): C4-filtered features (sparse, registration-only).
        # These must be separated because C4 yields too few points for reliable covariance
        # estimation in the VoxelMap (e.g. 450 vs 21k in GEODE tunnel → degenerate maps).
        _raw_xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        _raw_ints = np.ascontiguousarray(intensities, dtype=np.float64)
        _ranges = np.linalg.norm(_raw_xyz, axis=1)
        _map_mask = (_ranges > self.min_range) & (_ranges < self.max_range)
        _map_xyz = np.ascontiguousarray(_raw_xyz[_map_mask], dtype=np.float64)
        _map_ints = np.ascontiguousarray(_raw_ints[_map_mask], dtype=np.float64)
        if len(_map_ints) > 0:
            _p99 = float(np.percentile(_map_ints, 99))
            if _p99 > 1.0:
                _map_ints = _map_ints / _p99

        # Preprocessing: range filter + C4 downsampling for registration source
        points, intensities = self._prefilter(points, intensities)
        if len(points) < 10:
            pose = self.current_pose.copy()
            self.trajectory.poses.append(pose)
            self.trajectory.timestamps.append(timestamp or 0.0)
            return OdometryResult(pose=pose, timestamp=timestamp, odometry_mode="iv")

        # First frame: initialize map only (no registration target yet)
        if self._frame_count == 0:
            xyz = np.ascontiguousarray(_map_xyz, dtype=np.float64)
            ints_c = np.ascontiguousarray(_map_ints, dtype=np.float64)
            self._cpp_voxel_map.insert_frame(xyz, ints_c, 0)
            self._build_voxels_from_cpp_map()
            if self._cpp_coarse_map is not None:
                self._cpp_coarse_map.insert_frame(xyz, ints_c, 0)
                self._build_coarse_voxels()
            # C2 data-driven scale: alpha = 1/I_scale so residual ~ unit; uniform I → large alpha but omega_I small → cost small.
            if self._auto_alpha_from_intensity and self.iv_gicp.alpha > 0 and len(intensities) > 10:
                mad = np.median(np.abs(intensities - np.median(intensities)))
                i_scale = float(max(mad * 1.4826, 1e-9))
                self.iv_gicp.alpha = float(np.clip(1.0 / i_scale, 1e-6, 1e4))
            self._frame_count = 1
            self.trajectory.poses = [np.eye(4)]
            self.trajectory.timestamps = [timestamp or 0.0]
            return OdometryResult(pose=np.eye(4), timestamp=timestamp, odometry_mode="iv")

        # Initial pose: constant-velocity prediction (unless overridden)
        if init_pose is None:
            init_pose = self._predict_initial_pose()

        # [I1] Adaptive correspondence distance — KISS-ICP exact formula.
        # sigma = RMS of model deviations (pose prediction vs registration result).
        # adaptive_corr = 3σ (3-sigma rule, same as KISS-ICP).
        # Breaks the positive-feedback loop: sigma only grows from pose-prediction
        # errors (model deviation), NOT from correspondence distances.
        pred_motion = np.linalg.norm(init_pose[:3, 3] - self.current_pose[:3, 3])
        adaptive_corr = max(3.0 * self._model_dev_sigma, self._min_motion_th)
        adaptive_corr = min(adaptive_corr, self.iv_gicp.max_corr_dist)

        # Multi-scale: run coarse GICP first to improve initial pose estimate.
        # Coarse map has larger voxels → wider convergence basin → fewer local minima.
        # Result feeds as init_pose into the fine-level registration.
        if self._cpp_coarse_target_arrays is not None and self._coarse_iv_gicp is not None:
            T_coarse, _ = self._coarse_iv_gicp.register_with_arrays(
                points[:, :3],
                intensities,
                self._cpp_coarse_target_arrays,
                init_pose=init_pose,
                max_corr_dist=self._coarse_max_corr,
                session=None,
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
                self._build_voxels_from_cpp_map()
                self._last_auto_alpha = effective_alpha
                self._last_alpha_rebuild_frame = self._frame_count

        # Register source → local voxel map (C++)
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
            if mean_grad < self._intensity_gradient_threshold:
                effective_alpha = 0.0
        _alpha_restore = None
        if effective_alpha != self.iv_gicp.alpha:
            _alpha_restore = self.iv_gicp.alpha
            self.iv_gicp.alpha = effective_alpha
        _t_reg = time.perf_counter()
        if self._cpp_target_arrays is None:
            T = init_pose.copy() if init_pose is not None else self.current_pose.copy()
            reg_info = {"n_correspondences": 0, "converged": False}
        else:
            T, reg_info = self.iv_gicp.register_with_arrays(
                points[:, :3],
                intensities,
                self._cpp_target_arrays,
                init_pose=init_pose,
                max_corr_dist=adaptive_corr,
                session=getattr(self, "_registration_session", None),
                use_mscs=self._use_mscs,
                mscs_kappa_max=self._mscs_kappa_max,
                v_min_prev=self._prev_v_min,
            )
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
                            session=getattr(self, "_registration_session", None),
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
        # Use adaptive_corr (not fixed max_corr_dist) so threshold tightens when
        # the robot is moving slowly/steadily — prevents accepting large jumps.
        poor_registration = (n_corr < 10) or (result_dist > 2.0 * adaptive_corr)
        if poor_registration and self._frame_count > 1:
            T = init_pose.copy()

        # [I1] Update sigma: KISS-ICP cumulative SSE of model deviations.
        # sigma = sqrt(SSE / N) — converges to RMS prediction error.
        # Tight when tracking is consistent, wide when registration deviates.
        if not poor_registration and self._frame_count > 1:
            dev_T = se3_compose(se3_inverse(init_pose), T)  # T_pred^{-1} @ T_reg
            dev_xi = se3_log(dev_T)  # 6D Lie algebra deviation
            dev_norm = float(np.linalg.norm(dev_xi))
            self._model_error_sse2 += dev_norm ** 2
            self._model_n_samples += 1
            sigma_kiss = float(np.sqrt(self._model_error_sse2 / self._model_n_samples))
            self._model_dev_sigma = max(sigma_kiss, self._min_motion_th)

        # FIM proxy for next-frame plane/edge pooling (C4 source) + MSCS warm-start
        kappa_out = 0.0
        if not poor_registration:
            H_icp = reg_info.get("hessian")
            if H_icp is not None:
                Hm = np.asarray(H_icp, dtype=float)
                Hs = (Hm + Hm.T) * 0.5
                ev = np.linalg.eigvalsh(Hs)
                self._prev_h_min_eig = float(ev[0])
                kappa_out = float(ev[-1] / max(ev[0], 1e-9))
            else:
                self._prev_h_min_eig = None
            self._prev_n_corr = int(n_corr)
            # Update v_min warm-start for MSCS (from C++ result)
            v_min = reg_info.get("v_min")
            if v_min is not None:
                self._prev_v_min = np.asarray(v_min, dtype=np.float64)
        # else: keep previous _prev_* for stability

        self.current_pose = T
        self.trajectory.poses.append(self.current_pose.copy())
        self.trajectory.timestamps.append(timestamp or self.trajectory.timestamps[-1] + 1.0)

        # Only update the map when registration quality was sufficient.
        # Skipping map update on poor frames prevents corrupted points from
        # accumulating and causing cascade failures on subsequent frames.
        _t_map = time.perf_counter()
        if not poor_registration:
            self._update_map(_map_xyz, _map_ints, T, self._frame_count)
        map_ms = (time.perf_counter() - _t_map) * 1000
        self._frame_count += 1

        n_mscs_used = reg_info.get("n_mscs_used", 0)
        mscs_ratio = float(n_mscs_used) / max(n_corr, 1) if (self._use_mscs and n_corr > 0) else 1.0

        return OdometryResult(
            pose=T,
            timestamp=timestamp,
            num_correspondences=n_corr,
            reg_ms=reg_ms,
            map_ms=map_ms,
            odometry_mode="iv",
            kappa=kappa_out,
            mscs_ratio=mscs_ratio,
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

        self._cpp_voxel_map.insert_frame(pts_world, intensities, frame_id)
        if self._frame_count % 10 == 0:
            if self.map_radius is not None:
                cx, cy, cz = T[0, 3], T[1, 3], T[2, 3]
                self._cpp_voxel_map.evict_far_from(cx, cy, cz, self.map_radius)
            else:
                evict_before = frame_id - self._map_frames
                if evict_before > 0:
                    self._cpp_voxel_map.evict_before(evict_before)
            if self._ghost_raycast_threshold > 0.0:
                sx, sy, sz = T[0, 3], T[1, 3], T[2, 3]
                self._cpp_voxel_map.evict_ghost_raycast(
                    sx, sy, sz, pts_world,
                    ghost_threshold=self._ghost_raycast_threshold,
                    min_pass_through=self._ghost_min_pass_through,
                )

        if self._frame_count % self._kdtree_interval == 0 or self._cpp_target_arrays is None:
            self._build_voxels_from_cpp_map()

        if self._cpp_coarse_map is not None:
            self._cpp_coarse_map.insert_frame(pts_world, intensities, self._frame_count)
            if self._frame_count % 10 == 0:
                if self.map_radius is not None:
                    cx, cy, cz = T[0, 3], T[1, 3], T[2, 3]
                    self._cpp_coarse_map.evict_far_from(cx, cy, cz, self.map_radius)
                else:
                    evict_before_cc = self._frame_count - self._map_frames
                    if evict_before_cc > 0:
                        self._cpp_coarse_map.evict_before(evict_before_cc)
            if self._frame_count % self._kdtree_interval == 0 or self._cpp_coarse_target_arrays is None:
                self._build_coarse_voxels()

    def get_trajectory(self) -> Trajectory:
        return self.trajectory
