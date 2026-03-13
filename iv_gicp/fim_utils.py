"""
FIM Utilities: Unified Fisher Information Framework

Implements Section 11 (Theoretical Reinforcements) of IV-GICP paper:
  - Section 11.E: Unified FIM maximization framework
  - Section 11.F: Degeneracy Recovery Theorem (Theorem 1)

Well-posed regime (Theorem 1 converse, 2026):
  When I_G is already well-conditioned (κ_geo < τ), geometry-only is sufficient:
  v^T I_total v ≈ v^T I_G v > 0, so α→0 is theoretically justified. The pipeline
  option well_posed_polish refines the pose with 1–2 geometry-only GN steps when
  κ_geo < well_posed_kappa_threshold to reduce intensity noise in outdoor scenes.

Key functions:
  compute_fim_components()    -- split I_G and I_I from correspondences
  degeneracy_metrics()        -- λ_min, condition number, degeneracy index
  verify_degeneracy_recovery() -- numerical verification of Theorem 1
  fim_trace_summary()         -- unified FIM summary for all correspondences
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FIMComponents:
    """
    Split FIM into geometric and intensity components.

    Unified FIM (Section 11.E):
      I_total = I_G + I_I
      I_G = Σ_i J_xyz_i^T Ω_geo_i J_xyz_i   (6×6)
      I_I = Σ_i J_int_i^T ω_I_i J_int_i      (6×6)
    """
    I_G: np.ndarray          # (6,6) geometric FIM
    I_I: np.ndarray          # (6,6) intensity FIM
    I_total: np.ndarray      # (6,6) = I_G + I_I
    n_correspondences: int


@dataclass
class DegeneracyMetrics:
    """
    Per-direction degeneracy analysis for Theorem 1 verification.

    Theorem 1 states: if I_G is degenerate in direction v, then
      v^T I_total v >= (ε/σ_I²) Σ ||J_i v||² > 0
    """
    lambda_min_geo: float    # λ_min(I_G)
    lambda_min_total: float  # λ_min(I_total)
    condition_geo: float     # λ_max(I_G) / λ_min(I_G) -- inf if degenerate
    condition_total: float   # λ_max(I_total) / λ_min(I_total)
    degenerate_directions: np.ndarray  # (k, 6) eigenvectors of I_G with small eigenvalue
    recovery_values: np.ndarray        # v^T I_I v for each degenerate direction
    is_degenerate_geo: bool
    is_recovered_by_intensity: bool


# ─── Core FIM computation ─────────────────────────────────────────────────────

def _build_jacobians_batch(
    src_transformed: np.ndarray,   # (M, 3) transformed source points
    R_cur: np.ndarray,             # (3, 3) current rotation
    src_xyz: np.ndarray,           # (M, 3) source points (pre-transform)
    alpha: float,
    target_grads: np.ndarray,      # (M, 3) intensity gradients
) -> np.ndarray:
    """
    Build batch Jacobians J ∈ R^{M×4×6} for geo-photometric registration.

    J_xyz = [-[Rp]_×  |  I₃]     (3×6 geometric Jacobian)
    J_int = -α · ∇μ_I^T · J_xyz  (1×6 intensity Jacobian)
    J = [J_xyz; J_int]            (4×6 combined Jacobian)
    """
    M = len(src_xyz)
    I3 = np.eye(3)
    Rp = (R_cur @ src_xyz.T).T    # (M, 3)

    J_xyz = np.zeros((M, 3, 6))
    # Batch skew-symmetric: -[Rp]_×
    J_xyz[:, 0, 1] =  Rp[:, 2]
    J_xyz[:, 0, 2] = -Rp[:, 1]
    J_xyz[:, 1, 0] = -Rp[:, 2]
    J_xyz[:, 1, 2] =  Rp[:, 0]
    J_xyz[:, 2, 0] =  Rp[:, 1]
    J_xyz[:, 2, 1] = -Rp[:, 0]
    J_xyz[:, :, 3:6] = I3[np.newaxis, :, :]

    J = np.zeros((M, 4, 6))
    J[:, :3, :] = J_xyz
    # Intensity row: J[3,:] = -α · ∇μ_I^T · J_xyz
    J[:, 3, :] = -alpha * np.einsum('mi,mij->mj', target_grads, J_xyz)

    return J   # (M, 4, 6)


def compute_fim_components(
    src_xyz: np.ndarray,           # (M, 3) source points (already transformed)
    src_transformed: np.ndarray,   # (M, 3) = R @ src_xyz + t
    alpha: float,
    target_grads: np.ndarray,      # (M, 3) intensity gradients at target voxels
    omega_geo: np.ndarray,         # (M, 3, 3) per-correspondence geometric precision
    omega_int: np.ndarray,         # (M,)     per-correspondence intensity precision (scalar)
    R_cur: np.ndarray,             # (3, 3) current rotation (for Jacobian)
) -> FIMComponents:
    """
    Compute FIM split into geometric (I_G) and intensity (I_I) components.

    Unified FIM Framework (Section 11.E):
      I_total = I_G + I_I   maximized jointly by C1, C2, C3

    Args:
        src_xyz:         (M, 3) source points before transform
        src_transformed: (M, 3) source points after T applied (used for correspondence only)
        alpha:           intensity scaling constant
        target_grads:    (M, 3) ∇μ_I at matched target voxels
        omega_geo:       (M, 3, 3) Σ_geo^{-1} per correspondence
        omega_int:       (M,) 1/σ_I² per correspondence
        R_cur:           (3, 3) current rotation estimate

    Returns:
        FIMComponents with I_G, I_I, I_total all as (6,6) matrices
    """
    M = len(src_xyz)
    I3 = np.eye(3)
    Rp = (R_cur @ src_xyz.T).T    # (M, 3)

    # Build 3×6 geometric Jacobians
    J_xyz = np.zeros((M, 3, 6))
    J_xyz[:, 0, 1] =  Rp[:, 2]
    J_xyz[:, 0, 2] = -Rp[:, 1]
    J_xyz[:, 1, 0] = -Rp[:, 2]
    J_xyz[:, 1, 2] =  Rp[:, 0]
    J_xyz[:, 2, 0] =  Rp[:, 1]
    J_xyz[:, 2, 1] = -Rp[:, 0]
    J_xyz[:, :, 3:6] = I3[np.newaxis, :, :]

    # I_G = Σ_i J_xyz_i^T Ω_geo_i J_xyz_i
    # Shape: (M, 6, 3) @ (M, 3, 3) @ (M, 3, 6) -> (M, 6, 6) -> sum -> (6, 6)
    JtOmegaG = np.einsum('mji,mjk->mik', J_xyz, omega_geo)   # (M, 6, 3)
    I_G = np.einsum('mij,mjk->ik', JtOmegaG, J_xyz)           # (6, 6)

    # Build 1×6 intensity Jacobians
    J_int = -alpha * np.einsum('mi,mij->mj', target_grads, J_xyz)  # (M, 6)

    # I_I = Σ_i ω_I_i J_int_i^T J_int_i
    # Shape: (M, 6) scaled by omega_int -> (6, 6)
    I_I = np.einsum('m,mi,mj->ij', omega_int, J_int, J_int)   # (6, 6)

    I_total = I_G + I_I

    return FIMComponents(
        I_G=I_G,
        I_I=I_I,
        I_total=I_total,
        n_correspondences=M,
    )


# ─── Degeneracy analysis (Section 11.F) ───────────────────────────────────────

def degeneracy_metrics(
    fim: FIMComponents,
    degeneracy_threshold: float = 1e-3,
) -> DegeneracyMetrics:
    """
    Compute degeneracy metrics to verify Theorem 1.

    Theorem 1 (Degeneracy Recovery):
      If λ_min(I_G) ≈ 0 (geometric degeneracy),
      then λ_min(I_total) > 0 whenever ε > 0 in C_I.

    Args:
        fim: FIMComponents from compute_fim_components()
        degeneracy_threshold: eigenvalue below this → direction is degenerate

    Returns:
        DegeneracyMetrics with full degeneracy analysis
    """
    # Eigendecomposition of I_G
    evals_G, evecs_G = np.linalg.eigh(fim.I_G)   # ascending order
    evals_T, evecs_T = np.linalg.eigh(fim.I_total)

    lambda_min_G = float(evals_G[0])
    lambda_max_G = float(evals_G[-1])
    lambda_min_T = float(evals_T[0])
    lambda_max_T = float(evals_T[-1])

    eps = 1e-12
    cond_G = lambda_max_G / max(abs(lambda_min_G), eps)
    cond_T = lambda_max_T / max(abs(lambda_min_T), eps)

    # Identify degenerate directions of I_G
    deg_mask = np.abs(evals_G) < degeneracy_threshold
    degenerate_directions = evecs_G[:, deg_mask].T   # (k, 6)

    # For each degenerate direction v, compute v^T I_I v
    # Theorem 1: this must be > 0 if degeneracy is recovered
    if len(degenerate_directions) > 0:
        recovery_values = np.array([
            float(v @ fim.I_I @ v) for v in degenerate_directions
        ])
    else:
        recovery_values = np.array([])

    is_degenerate = lambda_min_G < degeneracy_threshold
    is_recovered = (
        len(recovery_values) == 0
        or bool(np.all(recovery_values > degeneracy_threshold))
    )

    return DegeneracyMetrics(
        lambda_min_geo=lambda_min_G,
        lambda_min_total=lambda_min_T,
        condition_geo=cond_G,
        condition_total=cond_T,
        degenerate_directions=degenerate_directions,
        recovery_values=recovery_values,
        is_degenerate_geo=is_degenerate,
        is_recovered_by_intensity=is_recovered,
    )


def verify_degeneracy_recovery(
    fim: FIMComponents,
    direction: Optional[np.ndarray] = None,
) -> dict:
    """
    Numerical verification of Theorem 1 for a specific direction.

    Theorem 1 bound:
      v^T I_total v >= (ε/σ_I²) Σ_i ||J_i v||²

    Args:
        fim: FIMComponents
        direction: (6,) test direction. If None, uses eigenvector of min eigenvalue of I_G.

    Returns:
        dict with:
          'v_geo_contribution':   v^T I_G v
          'v_int_contribution':   v^T I_I v
          'v_total_contribution': v^T I_total v
          'theorem1_holds':       bool, I_total contribution > 0
    """
    if direction is None:
        # Use most degenerate direction of I_G
        evals, evecs = np.linalg.eigh(fim.I_G)
        direction = evecs[:, 0]  # smallest eigenvalue eigenvector

    v = direction / (np.linalg.norm(direction) + 1e-12)

    v_geo = float(v @ fim.I_G @ v)
    v_int = float(v @ fim.I_I @ v)
    v_total = float(v @ fim.I_total @ v)

    return {
        "direction": v,
        "v_geo_contribution": v_geo,
        "v_int_contribution": v_int,
        "v_total_contribution": v_total,
        "theorem1_holds": v_total > 0,
        "intensity_recovery_ratio": v_int / max(abs(v_geo), 1e-12),
    }


# ─── FIM summary (Section 11.E narrative) ────────────────────────────────────

def fim_trace_summary(fim: FIMComponents) -> dict:
    """
    Summarize FIM contributions for the unified framework narrative.

    Section 11.E: tr(I_total) = tr(I_G) + tr(I_I)
    C1 maximizes tr(I_G) via voxel resolution.
    C2 fills null(I_G) via tr(I_I).
    C3 maintains map consistency (I_total stays valid after loop closure).
    """
    tr_G = float(np.trace(fim.I_G))
    tr_I = float(np.trace(fim.I_I))
    tr_total = float(np.trace(fim.I_total))

    evals_G = np.linalg.eigvalsh(fim.I_G)
    evals_T = np.linalg.eigvalsh(fim.I_total)

    return {
        "tr_I_geo": tr_G,
        "tr_I_int": tr_I,
        "tr_I_total": tr_total,
        "intensity_contribution_pct": 100.0 * tr_I / max(tr_total, 1e-12),
        "n_zero_eigenvalues_geo": int(np.sum(evals_G < 1e-3)),
        "n_zero_eigenvalues_total": int(np.sum(evals_T < 1e-3)),
        "lambda_min_geo": float(evals_G[0]),
        "lambda_min_total": float(evals_T[0]),
        "n_correspondences": fim.n_correspondences,
    }


# ─── Convenience: extract precision components from Voxel4D ──────────────────

def extract_precision_components(
    precision_4d: np.ndarray,   # (4, 4) combined precision matrix
) -> Tuple[np.ndarray, float]:
    """
    Split 4×4 combined precision matrix into geometric and intensity parts.

    The combined precision is built as:
      Ω = diag(Ω_geo, ω_I)   (block-diagonal, off-diag ≈ 0)

    Returns:
        omega_geo: (3, 3) geometric precision block
        omega_int: float, intensity precision scalar
    """
    omega_geo = precision_4d[:3, :3]
    omega_int = float(precision_4d[3, 3])
    return omega_geo, omega_int
