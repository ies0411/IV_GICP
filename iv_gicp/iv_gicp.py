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
    cov: np.ndarray            # (4,4) combined covariance (from information form)
    precision: np.ndarray = field(default_factory=lambda: np.eye(4))
    # (4,4) precomputed C^{-1} (avoids repeated pinv in GN loop)
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
        max_iterations: int = 32,
        convergence_threshold: float = 1e-6,
        min_voxel_size: float = 0.5,             # ℓ_v for σ_I² formula
        n_intensity_grad_neighbors: int = 8,     # K for gradient estimation
        device: str = "auto",                    # 'auto'|'cuda'|'cpu'|None
    ):
        self.alpha = alpha
        self.max_corr_dist = max_correspondence_distance
        self.max_iter = max_iterations
        self.conv_thresh = convergence_threshold
        self.min_voxel_size = min_voxel_size
        self.n_grad_nbrs = n_intensity_grad_neighbors
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

        # Remove self from each row and keep only n_grad_nbrs neighbors
        # Build clean (n, K) index array for batch computation
        K = self.n_grad_nbrs
        clean_idx = np.zeros((n, K), dtype=np.intp)
        valid_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            nbrs = all_nbr_idx[i]
            nbrs = nbrs[nbrs != i][:K]
            if len(nbrs) >= 3:
                # Pad with last neighbor if fewer than K
                if len(nbrs) < K:
                    nbrs = np.pad(nbrs, (0, K - len(nbrs)), mode='edge')
                clean_idx[i] = nbrs
                valid_mask[i] = True

        if not np.any(valid_mask):
            return

        # GPU batch lstsq for all valid voxels at once
        valid_i = np.where(valid_mask)[0]
        grads = batch_intensity_gradients(
            means_arr, intensities, clean_idx[valid_i], self._device
        )  # (n_valid, 3)

        for out_i, voxel_i in enumerate(valid_i):
            voxels_4d[voxel_i].intensity_gradient = grads[out_i]

    def _build_target_map(
        self,
        target_points: np.ndarray,
        target_intensities: np.ndarray,
    ) -> Tuple[np.ndarray, List[Voxel4D], FastKDTree]:
        """
        Build 4D voxel map from target point cloud.
        Returns: (voxel_centers: (M,3), voxels: List[Voxel4D], KDTree)
        """
        vs = self.min_voxel_size
        voxel_keys: dict = {}

        for i in range(len(target_points)):
            k = tuple((target_points[i, :3] / vs).astype(int))
            if k not in voxel_keys:
                voxel_keys[k] = []
            voxel_keys[k].append(i)

        # Collect per-voxel data before GPU batch ops
        voxel_data = []  # list of (mean_xyz, cov_3d, mean_i, var_i)
        for indices in voxel_keys.values():
            pts = target_points[indices, :3]
            ints = target_intensities[indices]
            mean_xyz = np.mean(pts, axis=0)
            if len(pts) > 1:
                centered = pts - mean_xyz
                cov_3d = (centered.T @ centered) / (len(pts) - 1)
                if cov_3d.ndim < 2 or cov_3d.shape != (3, 3):
                    cov_3d = np.eye(3) * 1e-4
            else:
                cov_3d = np.eye(3) * 1e-4
            cov_3d = cov_3d + 1e-6 * np.eye(3)
            mean_i = float(np.mean(ints))
            var_i = float(np.var(ints)) if len(ints) > 1 else 0.0
            voxel_data.append((mean_xyz, cov_3d, mean_i, var_i))

        nv = len(voxel_data)
        covs_arr  = np.array([d[1] for d in voxel_data])   # (nv, 3, 3)
        var_i_arr = np.array([d[3] for d in voxel_data])   # (nv,)
        means_arr_4d = np.array([
            [d[0][0], d[0][1], d[0][2], self.alpha * d[2]] for d in voxel_data
        ])   # (nv, 4)

        # Batch build all precision matrices (GPU or numpy)
        precisions = batch_precision_matrices(
            covs_arr, var_i_arr, self.min_voxel_size, self.alpha,
            device=self._device,
        )   # (nv, 4, 4)

        voxels_4d: List[Voxel4D] = []
        means: List[np.ndarray] = []
        for i, d in enumerate(voxel_data):
            mean_xyz, cov_3d, mean_i, var_i = d
            C_combined = build_combined_covariance(cov_3d, var_i, self.min_voxel_size, self.alpha)
            voxels_4d.append(Voxel4D(
                mean=means_arr_4d[i],
                cov=C_combined,
                precision=precisions[i],
            ))
            means.append(mean_xyz)

        means_arr = np.array(means)
        tree = FastKDTree(means_arr)

        # Precompute intensity gradients ∇μ_I at each voxel center
        # (used in intensity Jacobian: J[3,:] = -α · ∇μ_I^T · J_xyz)
        self._compute_intensity_gradients(means_arr, voxels_4d, tree)

        return means_arr, voxels_4d, tree

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
        n_target = len(target_means_4d)
        info = {"iterations": 0, "n_correspondences": 0, "converged": False}

        for iteration in range(self.max_iter):
            R_cur = T[:3, :3]
            t_cur = T[:3, 3]

            # 1. Transform all source points: (N, 3)
            src_transformed = (R_cur @ src_xyz.T).T + t_cur

            # 2. Batch KNN query
            dists, indices = tree_t.query(
                src_transformed, k=1, distance_upper_bound=self.max_corr_dist
            )

            # 3. Valid correspondence mask
            valid = (indices < n_target) & (dists <= self.max_corr_dist)
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

            H, b = gn_hessian_gradient(
                src_trans_valid, src_int_valid,
                t_means_g, t_prec_g, t_grad_g,
                R_cur, self.alpha, self._device,
            )

            # 8. Solve with LM damping: δ = -(H + λI)^{-1} b
            try:
                delta = np.linalg.solve(H + 1e-6 * np.eye(6), -b)
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
    ) -> np.ndarray:
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
        """
        target_means_4d, target_precisions, target_grads = self._precompute_target_arrays(target_voxels)
        T = np.eye(4) if init_pose is None else np.array(init_pose, dtype=float)
        T, _ = self._gauss_newton_vectorized(
            source_points[:, :3], source_intensities,
            target_means_3d, target_means_4d, target_precisions, target_grads,
            target_tree, T,
        )
        return T
