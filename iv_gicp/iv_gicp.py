"""
IV-GICP: Geo-Photometric Probabilistic Registration

4D space: p_i = [x, y, z, α·I]^T
Cost: Σ d_i^T Ω_i d_i  where Ω_i = C_G^{-1} + C_I^{-1} (information form)
- C_i^G: geometric covariance (VGICP-style, from voxel point distribution)
- C_i^I: photometric covariance; σ_I² = α² / (Var(I)/ℓ_v² + ε)
         High intensity gradient → small σ_I² → tight constraint
         Flat intensity region   → large σ_I² → uninformative (geometry dominates)

Key fixes vs. draft:
  - Intensity Jacobian J[3,:] now correctly derived (≠ 0 when ∇μ_I exists)
  - Photometric covariance direction corrected (inverse of gradient proxy)
  - Combined covariance uses information form (numerically stable)
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.spatial import cKDTree
from dataclasses import dataclass, field

from iv_gicp.fast_kdtree import FastKDTree
from iv_gicp.gpu_backend import (
    get_device,
    batch_intensity_gradients,
    batch_precision_matrices,
    gn_hessian_gradient,
    TargetGPUCache,
)

# C++ core: full GN loop in Eigen (no Python/CUDA overhead per iteration)
try:
    from iv_gicp.cpp import iv_gicp_core as _cpp_core
    _CPP_CORE_AVAILABLE = True
except ImportError:
    _cpp_core = None
    _CPP_CORE_AVAILABLE = False


# ─── Lie algebra helpers ──────────────────────────────────────────────────────

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3D vector."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map for SO(3) via Rodrigues formula."""
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        return np.eye(3)
    K = skew_symmetric(omega / angle)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def se3_exp(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SE(3) exponential map. xi = [ω, v] (6D). Returns (R, t)."""
    omega, v = xi[:3], xi[3:6]
    R = so3_exp(omega)
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        t = v.copy()
    else:
        J = (
            np.eye(3)
            + (1 - np.cos(angle)) / (angle**2) * skew_symmetric(omega)
            + (angle - np.sin(angle)) / (angle**3) * (skew_symmetric(omega) @ skew_symmetric(omega))
        )
        t = J @ v
    return R, t


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """T: (4,4), p: (3,) -> transformed point."""
    return T[:3, :3] @ p[:3] + T[:3, 3]


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Voxel4D:
    """4D voxel: geometry + intensity, with precomputed intensity gradient."""
    mean: np.ndarray           # (4,) [x, y, z, α·I]
    precision: np.ndarray = field(default_factory=lambda: np.eye(4))
    # (4,4) precomputed C^{-1} (avoids repeated pinv in GN loop)
    cov: Optional[np.ndarray] = None
    # (4,4) combined covariance — optional, unused in GN hot-path (only precision used)
    intensity_gradient: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # (3,) ∇μ_I at voxel center, estimated from K-NN finite differences


# ─── Covariance builders ──────────────────────────────────────────────────────

def build_photometric_sigma_sq(
    intensity_var: float,
    voxel_size: float,
    alpha: float,
    sigma_min: float = 1e-6,
    sigma_max: float = 1e6,
) -> float:
    """
    Photometric variance: σ_I² = α² / (Var(I)/ℓ_v² + ε)

    Interpretation:
      Var(I)/ℓ_v² ≈ |∇I|² (intensity gradient proxy within voxel)
      High gradient → small σ_I² → tight intensity constraint (informative)
      Near-zero gradient → σ_I² → σ_max → uninformative (geometry dominates)

    Args:
        intensity_var: within-voxel intensity variance (proxy for |∇I|²·ℓ_v²)
        voxel_size:    characteristic voxel side length ℓ_v [m]
        alpha:         intensity scaling constant (unit alignment)
    """
    eps_var = 1e-4  # prevents blow-up when intensity is perfectly uniform
    grad_sq_proxy = intensity_var / (voxel_size ** 2 + 1e-9)  # ≈ |∇I|²
    sigma_sq = alpha ** 2 / (grad_sq_proxy + eps_var)
    return float(np.clip(sigma_sq, sigma_min, sigma_max))


def build_combined_covariance(
    cov_3d: np.ndarray,
    intensity_var: float,
    voxel_size: float,
    alpha: float,
) -> np.ndarray:
    """
    Build 4×4 combined covariance via information form (numerically stable).

    C_combined = (C_G^{-1} + C_I^{-1})^{-1}

    where:
      C_G = diag(Σ_geo, σ_max)          geometric covariance (intensity block uninformative)
      C_I = diag(σ_max·I₃, σ_I²)        photometric covariance (geometric blocks uninformative)

    Information form:
      Ω = C_G^{-1} + C_I^{-1}
        ≈ diag(Σ_geo^{-1}, 0) + diag(0, 1/σ_I²)     (σ_max→∞ terms vanish)
        = diag(Σ_geo^{-1}, 1/σ_I²)

    This is exact in the limit σ_max → ∞ and avoids inverting ill-conditioned matrices.
    """
    eps_psd = 1e-6

    # Geometric Fisher information (3×3)
    Omega_geo = np.linalg.pinv(cov_3d + eps_psd * np.eye(3))

    # Photometric Fisher information (scalar)
    sigma_i_sq = build_photometric_sigma_sq(intensity_var, voxel_size, alpha)
    omega_i = 1.0 / (sigma_i_sq + eps_psd)

    # Full 4×4 information matrix
    Omega = np.zeros((4, 4))
    Omega[:3, :3] = Omega_geo
    Omega[3, 3] = omega_i

    # Invert to get covariance (block-diagonal structure → stable)
    C = np.linalg.pinv(Omega)
    return 0.5 * (C + C.T)  # enforce symmetry


# ─── IV-GICP class ────────────────────────────────────────────────────────────

class IVGICP:
    """
    Intensity-Augmented Voxelized GICP (IV-GICP).

    Registers source point cloud to a target map by minimizing:
      T* = argmin Σ_i d_i^T Ω_i d_i
    where d_i ∈ R^4 is the 4D geo-photometric residual and
    Ω_i = C_G^{-1} + C_I^{-1} is the combined precision matrix.

    The intensity Jacobian is correctly derived as:
      ∂d_I/∂ξ = -α · (∇μ_I)^T · J_xyz(p, T)
    where J_xyz = [-[Rp]_× | I₃] is the standard ICP pose Jacobian (3×6).
    """

    def __init__(
        self,
        alpha: float = 0.1,                      # intensity scaling (unit alignment: ~range/max_I)
        max_correspondence_distance: float = 2.0,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        min_voxel_size: float = 0.5,             # ℓ_v for σ_I² formula
        n_intensity_grad_neighbors: int = 8,     # K for gradient estimation
        huber_delta: float = 0.0,                # Huber threshold (0 = disabled)
        source_sigma: float = 0.0,               # source point uncertainty [m]; 0 = target-only Omega
        device: str = "auto",                    # 'auto'|'cuda'|'cpu'|None
    ):
        self.alpha = alpha
        self.max_corr_dist = max_correspondence_distance
        self.max_iter = max_iterations
        self.conv_thresh = convergence_threshold
        self.min_voxel_size = min_voxel_size
        self.n_grad_nbrs = n_intensity_grad_neighbors
        self.huber_delta = huber_delta
        self.source_sigma = source_sigma
        self._device = get_device(device)
        self._gpu_cache = TargetGPUCache(self._device)

    def _compute_intensity_gradients(
        self,
        means_arr: np.ndarray,
        voxels_4d: List[Voxel4D],
        tree: cKDTree,
    ) -> None:
        """
        Estimate intensity spatial gradient ∇μ_I at each voxel center
        via weighted least-squares over K nearest neighbors.

        For voxel i with neighbors {j}:
          A_k = c_j - c_i   (displacement, shape K×3)
          b_k = μ_I_j - μ_I_i  (intensity difference, shape K)
          grad_i = argmin ||A @ grad - b||²  (least squares)

        This approximates ∇μ_I to first order in the map's intensity field.
        """
        n = len(means_arr)
        if n < 4:
            return  # not enough neighbors; gradients remain zero

        k_query = min(self.n_grad_nbrs + 1, n)  # +1 because result includes self
        intensities = np.array([v.mean[3] / max(self.alpha, 1e-12) for v in voxels_4d])

        # Batch query (one call is faster than n individual queries)
        _, all_nbr_idx = tree.query(means_arr, k=k_query)  # (n, k_query)

        # Vectorized neighbor cleanup: index 0 is always self (querying cloud against itself).
        # Skip column 0, take next K columns; pad/clamp if fewer than K exist.
        K = self.n_grad_nbrs
        clean_idx = all_nbr_idx[:, 1:]                     # (n, k_query-1) — drop self
        k_got = clean_idx.shape[1]
        if k_got < K:
            # Pad by repeating last column
            pad = np.repeat(clean_idx[:, -1:], K - k_got, axis=1)
            clean_idx = np.concatenate([clean_idx, pad], axis=1)
        else:
            clean_idx = clean_idx[:, :K]
        clean_idx = np.minimum(clean_idx, n - 1).astype(np.intp)  # clamp out-of-bounds

        if clean_idx.shape[1] < 3:
            return

        grads = batch_intensity_gradients(
            means_arr, intensities, clean_idx, self._device
        )  # (n, 3)

        for i, v in enumerate(voxels_4d):
            v.intensity_gradient = grads[i]

    def _build_target_map(
        self,
        target_points: np.ndarray,
        target_intensities: np.ndarray,
    ) -> Tuple[np.ndarray, List[Voxel4D], FastKDTree]:
        """
        Build 4D voxel map from target point cloud.
        Returns: (voxel_centers: (M,3), voxels: List[Voxel4D], KDTree)

        Fully vectorized: replaces O(N) Python point loop with numpy sort+unique.
        Complexity: O(N log N) for sort, all stats via numpy scatter-add.
        For N=47k points → ~3-8ms vs ~200ms for the old Python loop.
        """
        vs = self.min_voxel_size
        pts = target_points[:, :3]
        ints = target_intensities

        # ── Step 1: Vectorized voxel assignment ──────────────────────────────
        # Map each point to integer voxel key, sort for contiguous grouping.
        keys_arr = np.floor(pts / vs).astype(np.int64)  # (N, 3)
        sort_idx = np.lexsort(keys_arr.T[::-1])          # sort by (x, y, z)
        keys_s   = keys_arr[sort_idx]                    # (N, 3) sorted keys
        pts_s    = pts[sort_idx]                         # (N, 3)
        ints_s   = ints[sort_idx]                        # (N,)

        _, inv_idx, counts = np.unique(
            keys_s, axis=0, return_inverse=True, return_counts=True
        )  # inv_idx: (N,), counts: (V,) — inv_idx is monotone since keys_s is sorted
        nv = len(counts)

        # Group boundaries for reduceat (fast: contiguous vs scatter)
        # inv_idx is sorted [0,0,...,1,1,...] so boundaries are change-points.
        bdry = np.concatenate([[0], np.where(np.diff(inv_idx))[0] + 1])  # (V,)

        # ── Step 2: Vectorized statistics via reduceat (cache-friendly) ───────
        # reduceat processes contiguous sorted groups — ~10x faster than add.at
        means = np.add.reduceat(pts_s, bdry) / counts[:, np.newaxis]     # (V, 3)

        centered = pts_s - means[inv_idx]                                 # (N, 3)
        outer    = (centered[:, :, np.newaxis] * centered[:, np.newaxis, :])  # (N,3,3)
        M2 = np.add.reduceat(outer.reshape(-1, 9), bdry).reshape(nv, 3, 3)   # (V,3,3)

        mean_i = np.add.reduceat(ints_s, bdry) / counts                  # (V,)
        ci     = ints_s - mean_i[inv_idx]
        var_i  = np.add.reduceat(ci ** 2, bdry)                           # (V,)

        n_safe = np.maximum(counts - 1, 1)
        covs  = M2 / n_safe[:, np.newaxis, np.newaxis]  # (V, 3, 3)
        covs += 1e-6 * np.eye(3)[np.newaxis]             # regularize
        var_i /= n_safe                                  # (V,)

        # ── Step 3: Filter to voxels with ≥ 1 point ──────────────────────────
        # Accept singletons: cov=1e-6·I (point-to-point ICP constraint).
        # Requiring ≥ 2 leaves only ~3-4 voxels for small clouds (e.g. 300 pts
        # with 0.5m voxels), causing GN divergence due to degenerate matches.
        valid    = counts >= 1
        means_v  = means[valid]    # (V', 3)
        covs_v   = covs[valid]     # (V', 3, 3)
        mean_i_v = mean_i[valid]   # (V',)
        var_i_v  = var_i[valid]    # (V',)
        nv_v = int(np.sum(valid))

        if nv_v == 0:
            return np.zeros((0, 3)), [], FastKDTree(np.zeros((1, 3)))

        # ── Step 4: Batch GPU precision matrices ──────────────────────────────
        precisions = batch_precision_matrices(
            covs_v, var_i_v, self.min_voxel_size, self.alpha,
            source_sigma=self.source_sigma,
            device=self._device,
        )  # (V', 4, 4)

        # ── Step 5: Build Voxel4D list (V' iterations, not N) ─────────────────
        # Note: Voxel4D.cov unused in GN hot-path (only .precision and .mean used).
        means_4d = np.column_stack([means_v, self.alpha * mean_i_v])  # (V', 4)
        voxels_4d = [
            Voxel4D(mean=means_4d[i], precision=precisions[i])
            for i in range(nv_v)
        ]

        tree = FastKDTree(means_v)
        self._compute_intensity_gradients(means_v, voxels_4d, tree)

        return means_v, voxels_4d, tree

    def register_with_arrays(
        self,
        src_xyz: np.ndarray,
        src_intensities: np.ndarray,
        target_arrays: dict,
        init_pose: Optional[np.ndarray] = None,
        max_corr_dist: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Register source to pre-built target arrays (C++ fast path).
        Bypasses Voxel4D extraction — arrays come directly from VoxelMap.build_target_arrays.
        """
        target_means_4d   = np.ascontiguousarray(target_arrays["means_4d"],  dtype=np.float64)
        target_precisions = np.ascontiguousarray(target_arrays["prec"],      dtype=np.float64)
        target_grads      = np.ascontiguousarray(target_arrays["grads"],     dtype=np.float64)
        means_3d          = np.ascontiguousarray(target_arrays["means_3d"],  dtype=np.float64)
        self._gpu_cache.load(target_means_4d, target_precisions, target_grads)

        T = np.eye(4) if init_pose is None else np.array(init_pose, dtype=float)
        # Build tree from means_3d for the Python GN fallback path.
        # The C++ GN path (iv_gicp_core) builds its own tree internally.
        tree = FastKDTree(means_3d)
        T, info = self._gauss_newton_vectorized(
            src_xyz[:, :3], src_intensities,
            means_3d, target_means_4d, target_precisions, target_grads,
            tree, T, max_corr_dist=max_corr_dist,
        )
        return T, info

    def _precompute_target_arrays(
        self,
        voxels_t: List[Voxel4D],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pre-extract target voxel data into contiguous arrays for vectorized GN.
        Also loads GPU cache so target arrays stay on GPU across GN iterations.

        Returns:
            target_means_4d: (M, 4) voxel means [x, y, z, α·I]
            target_precisions: (M, 4, 4) cached precision matrices C^{-1}
            target_grads: (M, 3) intensity gradients ∇μ_I
        """
        target_means_4d   = np.array([v.mean for v in voxels_t])              # (M, 4)
        target_precisions = np.array([v.precision for v in voxels_t])         # (M, 4, 4)
        target_grads      = np.array([v.intensity_gradient for v in voxels_t]) # (M, 3)
        # Pre-transfer to GPU once (reused for all GN iterations in this call)
        self._gpu_cache.load(target_means_4d, target_precisions, target_grads)
        return target_means_4d, target_precisions, target_grads

    def _gauss_newton_vectorized(
        self,
        src_xyz: np.ndarray,
        src_intensities: np.ndarray,
        means_t: np.ndarray,
        target_means_4d: np.ndarray,
        target_precisions: np.ndarray,
        target_grads: np.ndarray,
        tree_t: cKDTree,
        T: np.ndarray,
        max_corr_dist: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Vectorized Gauss-Newton optimization for geo-photometric registration.

        All per-point operations (residuals, Jacobians, Hessian accumulation) are
        batched via NumPy einsum, eliminating the Python per-point loop.

        Complexity per iteration: O(M) where M = valid correspondences,
        with NumPy-level vectorization (no Python loop over points).

        Args:
            src_xyz: (N, 3) source point coordinates
            src_intensities: (N,) source intensity values
            means_t: (M, 3) target voxel centers (for KDTree)
            target_means_4d: (M, 4) target voxel means [x,y,z,αI]
            target_precisions: (M, 4, 4) precomputed C^{-1}
            target_grads: (M, 3) intensity gradients
            tree_t: KDTree for nearest-voxel queries
            T: (4, 4) initial pose estimate

        Returns:
            T: (4, 4) optimized pose
            info: dict with convergence diagnostics
        """
        corr_dist = max_corr_dist if max_corr_dist is not None else self.max_corr_dist
        n_target = len(target_means_4d)
        info = {"iterations": 0, "n_correspondences": 0, "converged": False}
        H = None  # Hessian from last GN iteration (None if no correspondences found)

        # ── C++ fast path: full GN loop in Eigen (no Python/CUDA overhead) ───
        if _CPP_CORE_AVAILABLE:
            result = _cpp_core.icp_register(
                np.ascontiguousarray(src_xyz,          dtype=np.float64),
                np.ascontiguousarray(src_intensities,  dtype=np.float64),
                np.ascontiguousarray(target_means_4d,  dtype=np.float64),
                np.ascontiguousarray(target_precisions.reshape(n_target, 4, 4), dtype=np.float64),
                np.ascontiguousarray(target_grads,     dtype=np.float64),
                np.ascontiguousarray(means_t,          dtype=np.float64),
                np.ascontiguousarray(T,                dtype=np.float64),
                float(corr_dist),
                float(self.alpha),
                int(self.max_iter),
                float(self.huber_delta),
                6,  # min_valid
            )
            info["iterations"]       = result["iterations"]
            info["n_correspondences"] = result["n_valid"]
            info["converged"]        = result["converged"]
            if result["n_valid"] >= 6:
                info["hessian"] = result["H"]
            return result["T"], info

        # ── Python/GPU fallback (used if C++ not built) ───────────────────────
        for iteration in range(self.max_iter):
            R_cur = T[:3, :3]
            t_cur = T[:3, 3]

            # 1. Transform all source points: (N, 3)
            src_transformed = (R_cur @ src_xyz.T).T + t_cur

            # 2. Batch KNN query
            dists, indices = tree_t.query(
                src_transformed, k=1, distance_upper_bound=corr_dist
            )

            # 3. Valid correspondence mask
            valid = (indices < n_target) & (dists <= corr_dist)
            valid_idx = np.where(valid)[0]
            n_valid = len(valid_idx)
            if n_valid < 6:
                break

            info["n_correspondences"] = n_valid

            # 4. Gather all correspondence data (vectorized)
            src_valid = src_xyz[valid_idx]                    # (M', 3)
            src_trans_valid = src_transformed[valid_idx]      # (M', 3)
            src_int_valid = src_intensities[valid_idx]        # (M',)
            tidx = indices[valid_idx]                         # (M',)

            # 5-7. Residuals + Jacobians + Hessian (GPU or numpy)
            # target arrays gathered from GPU cache by correspondence indices
            t_means_g, t_prec_g, t_grad_g = self._gpu_cache.gather(tidx)

            # 8. Huber robust kernel: per-point downweighting of outlier correspondences.
            # w_i = 1 if dist_i < delta (inlier), delta/dist_i otherwise (outlier).
            # Passed into gn_hessian_gradient for exact per-point reweighting:
            #   H = Σ w_i * J_i^T Ω_i J_i,  b = Σ w_i * J_i^T Ω_i d_i
            if self.huber_delta > 0 and n_valid > 0:
                huber_d = dists[valid_idx]
                weights = np.where(huber_d < self.huber_delta,
                                   np.ones(n_valid),
                                   self.huber_delta / (huber_d + 1e-9))
            else:
                weights = None

            H, b = gn_hessian_gradient(
                src_trans_valid, src_int_valid,
                t_means_g, t_prec_g, t_grad_g,
                R_cur, self.alpha, self._device,
                weights=weights,
            )

            # 9. Solve: δ = -(H + λI)^{-1} b  with capped Marquardt damping.
            #
            # Problem with uncapped λ = 1e-4 × max_diag(H):
            #   KITTI outdoor: Ω_zz from ground voxels → max_diag(H) ~ 1e9,
            #   λ = 1e5.  Forward-direction H_xx ~ 5000 → step ≈ 5% GN.
            #   100 iters × 5% → 63% convergence only → significant residual drift.
            #
            # Fix: cap λ at 100 (near-GN for typical H_ii >> 100):
            #   H_xx=5000 → step = 98% GN  → converges in 2-3 iters  ✓
            #   H_zz=1e9  → step ≈ 100% GN → converges in 1 iter      ✓
            #   H_ii→0    → step bounded at b_ii/100                   ✓ (degenerate safe)
            try:
                lm_lambda = float(np.clip(1e-4 * np.max(np.abs(np.diag(H))), 1e-6, 100.0))
                delta = np.linalg.solve(H + lm_lambda * np.eye(6), -b)
            except np.linalg.LinAlgError:
                break

            # 9. Pose update: T ← exp(δ) · T  (left perturbation on SE(3))
            R_d, t_d = se3_exp(delta)
            T_d = np.eye(4)
            T_d[:3, :3] = R_d
            T_d[:3, 3] = t_d
            T = T_d @ T

            info["iterations"] = iteration + 1
            if np.linalg.norm(delta) < self.conv_thresh:
                info["converged"] = True
                break

        # Store the final Hessian for FORM-style window smoothing.
        # H approximates the Fisher information of the pose measurement:
        # high eigenvalues = well-constrained DOFs, low = degenerate DOFs.
        if H is not None:
            info["hessian"] = H
        return T, info

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        source_intensities: Optional[np.ndarray] = None,
        target_intensities: Optional[np.ndarray] = None,
        init_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Register source to target via geo-photometric Gauss-Newton optimization.
        Returns 4×4 transformation T such that T @ p_source ≈ p_target.

        Uses fully vectorized Gauss-Newton (NumPy einsum) for performance.
        """
        if source_points.shape[1] < 3:
            raise ValueError("Points must have at least 3 dimensions (x, y, z)")

        if source_intensities is None:
            source_intensities = (
                source_points[:, 3] if source_points.shape[1] >= 4
                else np.zeros(len(source_points))
            )
        if target_intensities is None:
            target_intensities = (
                target_points[:, 3] if target_points.shape[1] >= 4
                else np.zeros(len(target_points))
            )

        # Build 4D target voxel map with precomputed intensity gradients
        means_t, voxels_t, tree_t = self._build_target_map(target_points, target_intensities)

        # Pre-extract target arrays for vectorized GN
        target_means_4d, target_precisions, target_grads = self._precompute_target_arrays(voxels_t)

        T = np.eye(4) if init_pose is None else np.array(init_pose, dtype=float)

        T, _ = self._gauss_newton_vectorized(
            source_points[:, :3], source_intensities,
            means_t, target_means_4d, target_precisions, target_grads,
            tree_t, T,
        )
        return T

    def register_with_voxel_map(
        self,
        source_points: np.ndarray,
        source_intensities: np.ndarray,
        target_voxels: List[Voxel4D],
        target_means_3d: np.ndarray,
        target_tree: cKDTree,
        init_pose: Optional[np.ndarray] = None,
        max_corr_dist: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Register source to a pre-built voxel map (from AdaptiveVoxelMap).

        This avoids rebuilding the target voxel grid and allows adaptive
        voxel statistics (variable-resolution covariances) to be used directly.

        Args:
            source_points: (N, 3+) source scan
            source_intensities: (N,) intensity values
            target_voxels: pre-built Voxel4D list (from adaptive map)
            target_means_3d: (M, 3) voxel centers (for KDTree)
            target_tree: pre-built KDTree on target_means_3d
            init_pose: initial pose estimate
            max_corr_dist: override for correspondence distance (adaptive threshold)

        Returns:
            (T, info): optimized 4×4 pose and convergence diagnostics dict
        """
        target_means_4d, target_precisions, target_grads = self._precompute_target_arrays(target_voxels)
        T = np.eye(4) if init_pose is None else np.array(init_pose, dtype=float)
        T, info = self._gauss_newton_vectorized(
            source_points[:, :3], source_intensities,
            target_means_3d, target_means_4d, target_precisions, target_grads,
            target_tree, T,
            max_corr_dist=max_corr_dist,
        )
        return T, info
