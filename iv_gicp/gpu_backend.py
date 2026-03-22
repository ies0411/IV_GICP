"""
Optional PyTorch batch helpers for IV-GICP Python-side voxel prep (e.g. IVGICP.register).

  - batch_intensity_gradients
  - batch_precision_matrices

Registration itself runs in C++ (iv_gicp_core); these are numpy in → numpy out.
"""

import numpy as np
from typing import Optional, Union

# ─── Torch availability ───────────────────────────────────────────────────────

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_gpu_available() -> bool:
    """True if torch is installed and CUDA is accessible."""
    return _TORCH_AVAILABLE and torch.cuda.is_available()


def get_device(prefer: str = "auto") -> Optional[object]:
    """
    Resolve device string to torch.device (or None if torch unavailable).

    Args:
        prefer: 'auto' → cuda if available else cpu;
                'cuda' / 'gpu' → cuda (raise if not available);
                'cpu' → cpu torch;
                None  → None (use numpy fallback)
    """
    if not _TORCH_AVAILABLE:
        return None
    if prefer is None:
        return None
    if prefer in ("auto",):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    # Accept explicit torch.device objects
    return torch.device(prefer)


# ─── 1. Batch intensity gradient estimation ──────────────────────────────────


def batch_intensity_gradients(
    means_arr: np.ndarray,  # (n, 3) voxel centers
    intensities: np.ndarray,  # (n,)   voxel mean intensities
    all_nbr_idx: np.ndarray,  # (n, K) neighbor indices (from batch KDTree query)
    device,  # torch.device or None → numpy fallback
) -> np.ndarray:  # (n, 3) intensity gradients ∇μ_I
    """
    Estimate intensity spatial gradient at every voxel center via batch lstsq.

    Vectorized version of the per-voxel loop in IVGICP._compute_intensity_gradients():
      for each voxel i:
        A = means_arr[nbrs] - means_arr[i]    (K, 3)
        b = intensities[nbrs] - intensities[i] (K,)
        grad[i] = lstsq(A, b)

    Batched as: A_batch (n, K, 3), b_batch (n, K) → grad (n, 3)
    Uses torch.linalg.lstsq which dispatches to cuBLAS on GPU.

    Falls back to per-voxel numpy lstsq if device is None.
    """
    n, K = all_nbr_idx.shape

    # Build (n, K, 3) displacement and (n, K) intensity diff arrays
    nbr_xyz = means_arr[all_nbr_idx]  # (n, K, 3)
    nbr_int = intensities[all_nbr_idx]  # (n, K)
    A = nbr_xyz - means_arr[:, np.newaxis, :]  # (n, K, 3) displacements
    b = nbr_int - intensities[:, np.newaxis]  # (n, K) intensity diffs

    if device is None:
        # Numpy fallback: per-voxel lstsq
        grads = np.zeros((n, 3))
        for i in range(n):
            try:
                g, _, _, _ = np.linalg.lstsq(A[i], b[i], rcond=None)
                grads[i] = g
            except np.linalg.LinAlgError:
                pass
        return grads

    # GPU path: single batch lstsq call
    A_t = torch.tensor(A, dtype=torch.float64, device=device)  # (n, K, 3)
    b_t = torch.tensor(b, dtype=torch.float64, device=device)  # (n, K)

    # torch.linalg.lstsq expects (n, K, 3) and (n, K) → solution (n, 3)
    result = torch.linalg.lstsq(A_t, b_t.unsqueeze(-1)).solution  # (n, 3, 1)
    return result.squeeze(-1).cpu().numpy()  # (n, 3)


# ─── 2. Batch precision matrix construction ──────────────────────────────────


def batch_precision_matrices(
    covs_3d: np.ndarray,  # (n, 3, 3) geometric covariances
    var_intensities: np.ndarray,  # (n,)      within-voxel intensity variances
    voxel_sizes: Union[np.ndarray, float],  # (n,) or scalar effective voxel sizes
    alpha: float,
    epsilon: float = 1e-6,
    source_sigma: float = 0.0,  # source point position uncertainty [m]
    n_counts: Optional[np.ndarray] = None,  # (n,) point counts per voxel
    count_reg_scale: float = 2.0,  # prior stddev [m] for count-weighted regularization
    entropy_scale: Optional[np.ndarray] = None,  # (n,) C3: per-voxel scale for omega_I (high geo entropy -> scale > 1)
    device=None,
) -> np.ndarray:  # (n, 4, 4) precision matrices C^{-1}
    """
    Build all 4×4 precision matrices in a single batched GPU call.

    Used by IVGICP._build_target_map when building a target voxel grid in Python.

    Batched as:
      Omega (n, 4, 4) block-diagonal information matrix
      C_inv = torch.linalg.inv(Omega^{-1})  ← stable via information form

    Information form: Omega = diag(Sigma_combined^{-1}, omega_I)
      where Sigma_combined = Sigma_geo + source_sigma^2 * I  (standard GICP)
            omega_I = alpha² / (var_I / vsize² + eps_var)

    source_sigma > 0 adds a source-point noise floor to the target covariance,
    following the standard GICP combined covariance formulation (Segal 2009).
    This naturally bounds the precision matrices, preventing extreme anisotropy
    from degenerate voxels (e.g. flat ground planes: Omega_zz → 1/source_sigma²
    instead of 1/eps ≈ 1e6), which stabilizes the Gauss-Newton Hessian.
    """
    n = len(covs_3d)
    eps_psd = 1e-6
    eps_var = 1e-4

    if np.isscalar(voxel_sizes):
        voxel_sizes = np.full(n, float(voxel_sizes))

    # Build (n, 4, 4) information matrix Omega = diag(Omega_geo, omega_I)
    Omega = np.zeros((n, 4, 4), dtype=np.float64)

    # Geometric block: Omega_geo = (Sigma_geo + eps_psd*I + source_sigma^2*I + count_reg)^{-1}
    # Combined covariance (standard GICP: Sigma_source + Sigma_target).
    # source_sigma adds a noise floor that bounds Omega_geo eigenvalues to ≤ 1/source_sigma².
    Sigma_reg = covs_3d + eps_psd * np.eye(3)[np.newaxis]  # (n, 3, 3)
    if source_sigma > 0.0:
        Sigma_reg = Sigma_reg + (source_sigma**2) * np.eye(3)[np.newaxis]
    # Count-weighted regularization: Sigma += (count_reg_scale² / n) × I
    # For sparse voxels (small n): large isotropic term → behaves like point-to-point ICP.
    # For dense voxels (large n): negligible additive term → pure GICP covariance.
    # This bridges GICP and point-to-point, eliminating the accuracy cliff from sparse maps.
    if n_counts is not None:
        n_safe = np.maximum(n_counts, 1).astype(np.float64)  # (n,)
        count_reg = (count_reg_scale**2 / n_safe)[:, np.newaxis, np.newaxis] * np.eye(3)
        Sigma_reg = Sigma_reg + count_reg

    # Intensity precision scalar per voxel
    grad_sq_proxy = var_intensities / (voxel_sizes**2 + 1e-9)  # (n,)
    sigma_sq = np.clip(alpha**2 / (grad_sq_proxy + eps_var), 1e-6, 1e6)  # (n,)
    omega_I = 1.0 / (sigma_sq + eps_psd)  # (n,)
    if entropy_scale is not None:
        scale = np.asarray(entropy_scale, dtype=np.float64).reshape(-1)
        omega_I = omega_I * (scale ** 2)

    if device is None:
        # Omega IS the precision matrix: Omega = diag(Sigma_geo^{-1}, omega_I).
        # No double inversion needed — Omega is already in information form.
        for i in range(n):
            try:
                Omega[i, :3, :3] = np.linalg.inv(Sigma_reg[i])
            except np.linalg.LinAlgError:
                Omega[i, :3, :3] = np.linalg.pinv(Sigma_reg[i])
            Omega[i, 3, 3] = omega_I[i]
        return Omega

    # GPU path: Omega IS the precision matrix directly.
    Sigma_t = torch.tensor(Sigma_reg, dtype=torch.float64, device=device)  # (n, 3, 3)
    omega_I_t = torch.tensor(omega_I, dtype=torch.float64, device=device)  # (n,)

    # Batch invert geometric blocks: Omega_geo = Sigma_reg^{-1}
    try:
        Omega_geo = torch.linalg.inv(Sigma_t)  # (n, 3, 3)
    except Exception:
        Omega_geo = torch.linalg.pinv(Sigma_t)

    # Assemble (n, 4, 4) precision = diag(Omega_geo, omega_I)
    Omega_t = torch.zeros(n, 4, 4, dtype=torch.float64, device=device)
    Omega_t[:, :3, :3] = Omega_geo
    Omega_t[:, 3, 3] = omega_I_t

    return Omega_t.cpu().numpy()  # (n, 4, 4)
