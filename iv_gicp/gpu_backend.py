"""
GPU Backend for IV-GICP: PyTorch-accelerated batch operations.

Replaces the three main Python-loop bottlenecks with single GPU calls:
  1. batch_intensity_gradients()  — n×lstsq loops → 1 torch.linalg.lstsq call
  2. batch_precision_matrices()   — n×pinv loops  → 1 torch.linalg.inv call
  3. gn_hessian_gradient()        — numpy einsums → torch.einsum on GPU

Design principles:
  - float64 throughout (preserve accuracy identical to numpy)
  - numpy in → numpy out (transparent to callers)
  - graceful CPU fallback when torch is unavailable or device='cpu'
  - target arrays pre-transferred to GPU at map-build time (not per GN iteration)
"""

import numpy as np
from typing import Optional, Tuple, Union

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
    means_arr: np.ndarray,    # (n, 3) voxel centers
    intensities: np.ndarray,  # (n,)   voxel mean intensities
    all_nbr_idx: np.ndarray,  # (n, K) neighbor indices (from batch KDTree query)
    device,                   # torch.device or None → numpy fallback
) -> np.ndarray:              # (n, 3) intensity gradients ∇μ_I
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
    nbr_xyz = means_arr[all_nbr_idx]             # (n, K, 3)
    nbr_int = intensities[all_nbr_idx]           # (n, K)
    A = nbr_xyz - means_arr[:, np.newaxis, :]    # (n, K, 3) displacements
    b = nbr_int - intensities[:, np.newaxis]     # (n, K) intensity diffs

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
    A_t = torch.tensor(A, dtype=torch.float64, device=device)   # (n, K, 3)
    b_t = torch.tensor(b, dtype=torch.float64, device=device)   # (n, K)

    # torch.linalg.lstsq expects (n, K, 3) and (n, K) → solution (n, 3)
    result = torch.linalg.lstsq(A_t, b_t.unsqueeze(-1)).solution  # (n, 3, 1)
    return result.squeeze(-1).cpu().numpy()                         # (n, 3)


# ─── 2. Batch precision matrix construction ──────────────────────────────────

def batch_precision_matrices(
    covs_3d: np.ndarray,          # (n, 3, 3) geometric covariances
    var_intensities: np.ndarray,  # (n,)      within-voxel intensity variances
    voxel_sizes: Union[np.ndarray, float],  # (n,) or scalar effective voxel sizes
    alpha: float,
    epsilon: float = 1e-6,
    device=None,
) -> np.ndarray:                  # (n, 4, 4) precision matrices C^{-1}
    """
    Build all 4×4 precision matrices in a single batched GPU call.

    Replaces the per-leaf loop in pipeline._build_voxels_from_adaptive_map():
      for leaf in leaves:
        C = build_combined_covariance(cov, var_I, vsize, alpha)
        C_inv = np.linalg.pinv(C + 1e-6 * I)

    Batched as:
      Omega (n, 4, 4) block-diagonal information matrix
      C_inv = torch.linalg.inv(Omega^{-1})  ← stable via information form

    Information form: Omega = diag(Sigma_geo^{-1}, omega_I)
      where omega_I = alpha² / (var_I / vsize² + eps_var)
    This is block-diagonal so inversion is exact and numerically stable.
    """
    n = len(covs_3d)
    eps_psd = 1e-6
    eps_var = 1e-4

    if np.isscalar(voxel_sizes):
        voxel_sizes = np.full(n, float(voxel_sizes))

    # Build (n, 4, 4) information matrix Omega = diag(Omega_geo, omega_I)
    Omega = np.zeros((n, 4, 4), dtype=np.float64)

    # Geometric block: Omega_geo = (Sigma_geo + eps I)^{-1}
    # Regularize first, then invert geometrically below
    Sigma_reg = covs_3d + eps_psd * np.eye(3)[np.newaxis]   # (n, 3, 3)

    # Intensity precision scalar per voxel
    grad_sq_proxy = var_intensities / (voxel_sizes ** 2 + 1e-9)   # (n,)
    sigma_sq = np.clip(alpha ** 2 / (grad_sq_proxy + eps_var), 1e-6, 1e6)  # (n,)
    omega_I = 1.0 / (sigma_sq + eps_psd)                                   # (n,)

    if device is None:
        # Numpy fallback: per-matrix inversion
        for i in range(n):
            try:
                Omega[i, :3, :3] = np.linalg.inv(Sigma_reg[i])
            except np.linalg.LinAlgError:
                Omega[i, :3, :3] = np.linalg.pinv(Sigma_reg[i])
            Omega[i, 3, 3] = omega_I[i]

        # Omega is block-diagonal, so C = Omega^{-1} is also block-diagonal
        C = np.zeros((n, 4, 4), dtype=np.float64)
        for i in range(n):
            try:
                C[i] = np.linalg.inv(Omega[i] + eps_psd * np.eye(4))
            except np.linalg.LinAlgError:
                C[i] = np.linalg.pinv(Omega[i] + eps_psd * np.eye(4))
        # symmetrize
        C = 0.5 * (C + C.transpose(0, 2, 1))
        # C_inv = Omega (since C = Omega^{-1} → C^{-1} = Omega)
        prec = np.zeros((n, 4, 4), dtype=np.float64)
        for i in range(n):
            try:
                prec[i] = np.linalg.inv(C[i] + eps_psd * np.eye(4))
            except np.linalg.LinAlgError:
                prec[i] = np.linalg.pinv(C[i] + eps_psd * np.eye(4))
        return prec

    # GPU path: batch inversion
    Sigma_t = torch.tensor(Sigma_reg, dtype=torch.float64, device=device)    # (n, 3, 3)
    omega_I_t = torch.tensor(omega_I, dtype=torch.float64, device=device)    # (n,)
    eps_t = torch.tensor(eps_psd, dtype=torch.float64, device=device)

    # Batch invert geometric blocks
    try:
        Omega_geo = torch.linalg.inv(Sigma_t)   # (n, 3, 3)
    except Exception:
        Omega_geo = torch.linalg.pinv(Sigma_t)

    # Assemble (n, 4, 4) Omega
    Omega_t = torch.zeros(n, 4, 4, dtype=torch.float64, device=device)
    Omega_t[:, :3, :3] = Omega_geo
    Omega_t[:, 3, 3] = omega_I_t

    # C = Omega^{-1}  (block-diagonal → stable)
    Omega_reg = Omega_t + eps_t * torch.eye(4, dtype=torch.float64, device=device)
    try:
        C_t = torch.linalg.inv(Omega_reg)
    except Exception:
        C_t = torch.linalg.pinv(Omega_reg)

    # Symmetrize C
    C_t = 0.5 * (C_t + C_t.transpose(-1, -2))

    # Precision = C^{-1}
    C_reg = C_t + eps_t * torch.eye(4, dtype=torch.float64, device=device)
    try:
        prec_t = torch.linalg.inv(C_reg)
    except Exception:
        prec_t = torch.linalg.pinv(C_reg)

    return prec_t.cpu().numpy()   # (n, 4, 4)


# ─── 3. GPU-accelerated GN Hessian/gradient accumulation ─────────────────────

def gn_hessian_gradient(
    src_valid: np.ndarray,         # (M', 3) transformed source points
    src_int_valid: np.ndarray,     # (M',)   source intensities
    t_means_gpu,                   # (M', 4) GPU tensor (pre-cached target means)
    t_prec_gpu,                    # (M', 4, 4) GPU tensor (pre-cached precisions)
    t_grad_gpu,                    # (M', 3) GPU tensor (pre-cached intensity grads)
    R_cur: np.ndarray,             # (3, 3) current rotation
    alpha: float,
    device,
) -> Tuple[np.ndarray, np.ndarray]:   # H (6,6), b (6,)
    """
    GPU-accelerated Gauss-Newton Hessian and gradient accumulation.

    Replaces the three einsum calls in _gauss_newton_vectorized():
      JtC = einsum('mji,mjk->mik', J, t_prec)   (M', 6, 4)
      H   = einsum('mij,mjk->ik',  JtC, J)       (6, 6)
      b   = einsum('mij,mj->i',    JtC, d)       (6,)

    Target arrays (t_means, t_prec, t_grad) are pre-cached on GPU and indexed
    by valid correspondence indices to avoid per-iteration full transfers.

    Falls back to numpy if device is None (t_means_gpu etc. are then numpy arrays).
    """
    if device is None:
        # Pure numpy fallback
        # Ensure inputs are numpy for fallback
        def _to_np(x):
            if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.asarray(x)
        return _gn_hessian_numpy(src_valid, src_int_valid,
                                 _to_np(t_means_gpu), _to_np(t_prec_gpu), _to_np(t_grad_gpu),
                                 R_cur, alpha)

    M = len(src_valid)
    I3 = torch.eye(3, dtype=torch.float64, device=device)

    # Transfer small per-iteration arrays (source side only)
    src_t = torch.tensor(np.asarray(src_valid), dtype=torch.float64, device=device)
    src_i = torch.tensor(np.asarray(src_int_valid), dtype=torch.float64, device=device)
    R_t   = torch.tensor(np.asarray(R_cur), dtype=torch.float64, device=device)

    # Ensure target arrays are torch tensors on the correct device
    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float64)
        return torch.tensor(np.asarray(x), dtype=torch.float64, device=device)

    t_means = _to_torch(t_means_gpu)   # (M', 4)
    t_prec  = _to_torch(t_prec_gpu)    # (M', 4, 4)
    t_grad  = _to_torch(t_grad_gpu)    # (M', 3)

    # 4D Residual
    d_xyz = src_t - t_means[:, :3]                        # (M', 3)
    d_i   = alpha * src_i - t_means[:, 3]                 # (M',)
    d     = torch.cat([d_xyz, d_i.unsqueeze(-1)], dim=-1) # (M', 4)

    # Jacobian J_xyz (M', 3, 6)
    Rp = (R_t @ src_t.T).T     # (M', 3)
    J_xyz = torch.zeros(M, 3, 6, dtype=torch.float64, device=device)
    J_xyz[:, 0, 1] =  Rp[:, 2]
    J_xyz[:, 0, 2] = -Rp[:, 1]
    J_xyz[:, 1, 0] = -Rp[:, 2]
    J_xyz[:, 1, 2] =  Rp[:, 0]
    J_xyz[:, 2, 0] =  Rp[:, 1]
    J_xyz[:, 2, 1] = -Rp[:, 0]
    J_xyz[:, :, 3:6] = I3.unsqueeze(0)                    # broadcast (1,3,3)

    # Full Jacobian J (M', 4, 6)
    J = torch.zeros(M, 4, 6, dtype=torch.float64, device=device)
    J[:, :3, :] = J_xyz
    J[:, 3, :]  = -alpha * torch.einsum('mi,mij->mj', t_grad, J_xyz)

    # Hessian accumulation
    JtC = torch.einsum('mji,mjk->mik', J, t_prec)  # (M', 6, 4)
    H   = torch.einsum('mij,mjk->ik',  JtC, J)     # (6, 6)
    b   = torch.einsum('mij,mj->i',    JtC, d)     # (6,)

    return H.cpu().numpy(), b.cpu().numpy()


def _gn_hessian_numpy(
    src_valid, src_int_valid,
    t_means, t_prec, t_grad,
    R_cur, alpha,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure numpy fallback for gn_hessian_gradient (identical math)."""
    M = len(src_valid)
    I3 = np.eye(3)

    d_xyz = src_valid - t_means[:, :3]
    d_i   = alpha * src_int_valid - t_means[:, 3]
    d     = np.column_stack([d_xyz, d_i])

    Rp = (R_cur @ src_valid.T).T
    J_xyz = np.zeros((M, 3, 6))
    J_xyz[:, 0, 1] =  Rp[:, 2]
    J_xyz[:, 0, 2] = -Rp[:, 1]
    J_xyz[:, 1, 0] = -Rp[:, 2]
    J_xyz[:, 1, 2] =  Rp[:, 0]
    J_xyz[:, 2, 0] =  Rp[:, 1]
    J_xyz[:, 2, 1] = -Rp[:, 0]
    J_xyz[:, :, 3:6] = I3[np.newaxis]

    J = np.zeros((M, 4, 6))
    J[:, :3, :] = J_xyz
    J[:, 3, :]  = -alpha * np.einsum('mi,mij->mj', t_grad, J_xyz)

    JtC = np.einsum('mji,mjk->mik', J, t_prec)
    H   = np.einsum('mij,mjk->ik',  JtC, J)
    b   = np.einsum('mij,mj->i',    JtC, d)
    return H, b


# ─── 4. GPU cache helpers ─────────────────────────────────────────────────────

class TargetGPUCache:
    """
    Holds target voxel arrays on GPU between GN iterations.

    Workflow:
      cache = TargetGPUCache(device)
      cache.load(means_4d, precisions, grads)   # once per map rebuild
      ...
      # Inside GN loop (per iteration):
      t_means, t_prec, t_grad = cache.gather(tidx)  # index into GPU arrays
    """

    def __init__(self, device):
        self.device = device
        self._means_gpu  = None   # (M_full, 4)
        self._prec_gpu   = None   # (M_full, 4, 4)
        self._grads_gpu  = None   # (M_full, 3)

    def load(
        self,
        means_4d:    np.ndarray,   # (M, 4)
        precisions:  np.ndarray,   # (M, 4, 4)
        grads:       np.ndarray,   # (M, 3)
    ) -> None:
        """Transfer full target arrays to GPU. Call once per map rebuild."""
        if self.device is None:
            self._means_gpu  = means_4d
            self._prec_gpu   = precisions
            self._grads_gpu  = grads
            return
        self._means_gpu = torch.tensor(means_4d,   dtype=torch.float64, device=self.device)
        self._prec_gpu  = torch.tensor(precisions, dtype=torch.float64, device=self.device)
        self._grads_gpu = torch.tensor(grads,      dtype=torch.float64, device=self.device)

    def gather(self, tidx: np.ndarray):
        """
        Index into GPU arrays by correspondence indices.
        Returns (means, prec, grads) as GPU tensors (or numpy if device=None).
        """
        if self.device is None:
            return (self._means_gpu[tidx],
                    self._prec_gpu[tidx],
                    self._grads_gpu[tidx])
        tidx_t = torch.tensor(tidx, dtype=torch.long, device=self.device)
        return (self._means_gpu[tidx_t],
                self._prec_gpu[tidx_t],
                self._grads_gpu[tidx_t])

    @property
    def is_loaded(self) -> bool:
        return self._means_gpu is not None
