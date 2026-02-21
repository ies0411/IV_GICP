"""
Fisher Information Matrix (FIM) Degeneracy Analysis for IV-GICP.

This module implements the theoretical core of the IV-GICP paper:
Proposition 1 (FIM Complementarity) and runtime degeneracy monitoring.

Proposition 1 (FIM Complementarity):
  Let F = F_geo + F_photo be the combined 6×6 Fisher Information Matrix,
  where F_geo accumulates geometric constraints and F_photo accumulates
  photometric (intensity) constraints. Suppose F_geo has a rank deficiency
  along direction v ∈ R^6 (i.e., v^T F_geo v < ε for unit v). Then F is
  full-rank if and only if there exists at least one correspondence i whose
  intensity gradient satisfies:
    (∇μ_I)^T · J_xyz(p_i, T) · v ≠ 0

  Proof sketch:
    The geometric FIM is F_geo = Σ_i J_xyz_i^T Ω_geo_i J_xyz_i.
    The photometric FIM is F_photo = Σ_i (1/σ_I_i²) · J_I_i^T J_I_i
    where J_I_i = -α · (∇μ_I_i)^T · J_xyz_i  (intensity Jacobian, 1×6).

    For v in null(F_geo):
      v^T F_photo v = Σ_i (1/σ_I_i²) · ((∇μ_I_i)^T J_xyz_i v)²
    This is positive iff ∃i: (∇μ_I_i)^T J_xyz_i v ≠ 0.

    Physical interpretation: Intensity gradients that have a component along
    the geometrically degenerate direction provide constraint. For example,
    in a long corridor (degenerate along corridor axis), lane markings on
    the floor provide intensity gradients perpendicular to the markings,
    which typically have a component along the corridor axis.

Usage:
  After IV-GICP registration, call analyze_registration_degeneracy() with
  the Gauss-Newton intermediates to generate FIM analysis for the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DegeneracyReport:
    """Result of FIM degeneracy analysis for a single registration."""
    combined_eigenvalues: np.ndarray    # (6,) eigenvalues of F_combined, ascending
    geometric_eigenvalues: np.ndarray   # (6,) eigenvalues of F_geo
    photometric_eigenvalues: np.ndarray # (6,) eigenvalues of F_photo
    condition_number: float             # cond(F_combined)
    geo_condition_number: float         # cond(F_geo)
    is_geo_degenerate: bool             # True if min eigenvalue of F_geo < threshold
    is_rescued: bool                    # True if F_photo fills the degenerate direction
    degenerate_directions: np.ndarray   # (k, 6) eigenvectors of degenerate directions
    rescue_strengths: np.ndarray        # (k,) v^T F_photo v for each degenerate direction
    n_correspondences: int


def compute_geometric_fim(
    src_xyz_transformed: np.ndarray,
    R: np.ndarray,
    precisions_geo_3x3: np.ndarray,
) -> np.ndarray:
    """
    Compute geometric Fisher Information Matrix (6×6).

    F_geo = Σ_i J_xyz_i^T Ω_geo_i J_xyz_i

    where J_xyz_i = [-[Rp_i]_× | I₃] is the standard ICP Jacobian (3×6)
    and Ω_geo_i is the 3×3 geometric precision matrix.

    Args:
        src_xyz_transformed: (M, 3) source points after transformation
        R: (3, 3) current rotation estimate
        precisions_geo_3x3: (M, 3, 3) geometric precision matrices (Σ_geo^{-1})

    Returns:
        F_geo: (6, 6) geometric FIM
    """
    M = len(src_xyz_transformed)
    Rp = src_xyz_transformed  # already = R @ p_src + t

    # Build batch J_xyz: (M, 3, 6)
    J_xyz = np.zeros((M, 3, 6))
    J_xyz[:, 0, 1] = Rp[:, 2]
    J_xyz[:, 0, 2] = -Rp[:, 1]
    J_xyz[:, 1, 0] = -Rp[:, 2]
    J_xyz[:, 1, 2] = Rp[:, 0]
    J_xyz[:, 2, 0] = Rp[:, 1]
    J_xyz[:, 2, 1] = -Rp[:, 0]
    J_xyz[:, :, 3:6] = np.eye(3)[np.newaxis, :, :]

    # F_geo = Σ J^T Ω J via einsum
    # JtOmega: (M, 6, 3) = J^T @ Omega
    JtOmega = np.einsum('mji,mjk->mik', J_xyz, precisions_geo_3x3)
    # F_geo: (6, 6) = Σ JtOmega @ J
    F_geo = np.einsum('mij,mjk->ik', JtOmega, J_xyz)

    return F_geo


def compute_photometric_fim(
    src_xyz_transformed: np.ndarray,
    R: np.ndarray,
    intensity_gradients: np.ndarray,
    sigma_I_sq: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute photometric Fisher Information Matrix (6×6).

    F_photo = Σ_i (1/σ_I_i²) · J_I_i^T J_I_i

    where J_I_i = -α · (∇μ_I_i)^T · J_xyz_i  is the intensity Jacobian (1×6).

    The photometric FIM captures constraint from intensity texture. In featureless
    environments (uniform intensity), all σ_I² → ∞ and F_photo → 0.
    In textured regions (lane markings, wall textures), σ_I² is small and
    F_photo provides significant constraint.

    Args:
        src_xyz_transformed: (M, 3) transformed source points
        R: (3, 3) rotation estimate
        intensity_gradients: (M, 3) ∇μ_I at each target voxel
        sigma_I_sq: (M,) photometric variance per correspondence
        alpha: intensity scaling constant

    Returns:
        F_photo: (6, 6) photometric FIM
    """
    M = len(src_xyz_transformed)
    Rp = src_xyz_transformed

    # Build J_xyz: (M, 3, 6)
    J_xyz = np.zeros((M, 3, 6))
    J_xyz[:, 0, 1] = Rp[:, 2]
    J_xyz[:, 0, 2] = -Rp[:, 1]
    J_xyz[:, 1, 0] = -Rp[:, 2]
    J_xyz[:, 1, 2] = Rp[:, 0]
    J_xyz[:, 2, 0] = Rp[:, 1]
    J_xyz[:, 2, 1] = -Rp[:, 0]
    J_xyz[:, :, 3:6] = np.eye(3)[np.newaxis, :, :]

    # J_I = -alpha * grad^T @ J_xyz : (M, 6)
    J_I = -alpha * np.einsum('mi,mij->mj', intensity_gradients, J_xyz)

    # Precision weights: (M,)
    omega_I = 1.0 / (sigma_I_sq + 1e-12)

    # F_photo = Σ omega_I_i * J_I_i^T @ J_I_i : (6, 6)
    # = Σ omega_I_i * outer(J_I_i, J_I_i)
    weighted_J = J_I * np.sqrt(omega_I)[:, np.newaxis]  # (M, 6)
    F_photo = weighted_J.T @ weighted_J  # (6, 6)

    return F_photo


def analyze_degeneracy(
    F_geo: np.ndarray,
    F_photo: np.ndarray,
    degeneracy_threshold: float = 1e-3,
    n_correspondences: int = 0,
) -> DegeneracyReport:
    """
    Analyze FIM degeneracy and photometric rescue capability.

    Steps:
      1. Eigendecompose F_geo to find degenerate directions (small eigenvalues)
      2. For each degenerate direction v, compute v^T F_photo v
      3. If this is positive, intensity rescues the degeneracy

    Args:
        F_geo: (6, 6) geometric FIM
        F_photo: (6, 6) photometric FIM
        degeneracy_threshold: eigenvalue below this is considered degenerate
        n_correspondences: number of correspondences (for reporting)

    Returns:
        DegeneracyReport with full analysis
    """
    F_combined = F_geo + F_photo

    # Eigendecompose (ascending order)
    eig_combined, _ = np.linalg.eigh(F_combined)
    eig_geo, vec_geo = np.linalg.eigh(F_geo)
    eig_photo, _ = np.linalg.eigh(F_photo)

    # Condition numbers
    cond_combined = eig_combined[-1] / max(eig_combined[0], 1e-15)
    cond_geo = eig_geo[-1] / max(eig_geo[0], 1e-15)

    # Find degenerate directions
    degenerate_mask = eig_geo < degeneracy_threshold
    degenerate_dirs = vec_geo[:, degenerate_mask].T  # (k, 6)
    is_geo_degenerate = np.any(degenerate_mask)

    # Check photometric rescue: v^T F_photo v for each degenerate direction
    rescue_strengths = np.array([
        v @ F_photo @ v for v in degenerate_dirs
    ]) if len(degenerate_dirs) > 0 else np.array([])

    is_rescued = is_geo_degenerate and np.all(rescue_strengths > degeneracy_threshold)

    return DegeneracyReport(
        combined_eigenvalues=eig_combined,
        geometric_eigenvalues=eig_geo,
        photometric_eigenvalues=eig_photo,
        condition_number=float(cond_combined),
        geo_condition_number=float(cond_geo),
        is_geo_degenerate=bool(is_geo_degenerate),
        is_rescued=bool(is_rescued),
        degenerate_directions=degenerate_dirs,
        rescue_strengths=rescue_strengths,
        n_correspondences=n_correspondences,
    )


def check_photometric_rescue(
    F_geo: np.ndarray,
    F_photo: np.ndarray,
    threshold: float = 1e-3,
) -> bool:
    """
    Quick check: does photometric FIM fill the null space of geometric FIM?

    Returns True if F_combined = F_geo + F_photo is full-rank even when
    F_geo alone is not. This is the core claim of Proposition 1.
    """
    eig_geo = np.linalg.eigvalsh(F_geo)
    eig_combined = np.linalg.eigvalsh(F_geo + F_photo)

    geo_degenerate = eig_geo[0] < threshold
    combined_ok = eig_combined[0] > threshold

    return geo_degenerate and combined_ok


def genz_icp_condition_number(hessian_6x6: np.ndarray) -> float:
    """
    Compute GenZ-ICP style degeneracy metric.

    GenZ-ICP (Lee et al., RA-L 2025) detects degeneracy using the condition
    number of the translational block of the Hessian matrix:
        κ = sqrt(λ_max / λ_min)  of  H[:3, :3]

    A high κ indicates the translational optimization is ill-conditioned
    (e.g., in a long corridor where forward translation is unconstrained).

    Comparison with IV-GICP:
    - GenZ-ICP: computes κ → if κ > threshold, manually flags axis as degenerate
      and applies heuristic adaptive weighting between point-to-plane/point-to-point
    - IV-GICP: Theorem 1 guarantees F_combined is always positive definite (ε > 0),
      no explicit degeneracy detector or threshold needed.

    This function enables direct experimental comparison between the two approaches.

    Args:
        hessian_6x6: (6, 6) Gauss-Newton Hessian H = J^T Ω J in se(3) parameterization
                     (first 3 rows/cols = rotation, last 3 = translation in our layout;
                      or swap if using [t, φ] layout — pass the translational block's
                      3×3 sub-matrix extracted as hessian_6x6[:3, :3] or [:3, :3])

    Returns:
        Condition number κ = sqrt(λ_max / λ_min) of the 3×3 translational block.
        Returns np.inf if λ_min ≤ 0.
    """
    H_trans = hessian_6x6[3:, 3:]   # translation block (last 3×3 in [rot|trans] layout)
    eigs = np.linalg.eigvalsh(H_trans)
    lam_min = eigs[0]
    lam_max = eigs[-1]
    if lam_min <= 0:
        return np.inf
    return float(np.sqrt(lam_max / lam_min))


def compare_degeneracy_metrics(
    F_geo: np.ndarray,
    F_photo: np.ndarray,
    hessian_6x6: Optional[np.ndarray] = None,
    degeneracy_threshold: float = 1e-3,
    genz_kappa_threshold: float = 100.0,
) -> Dict:
    """
    Side-by-side comparison of IV-GICP vs GenZ-ICP degeneracy metrics.

    Produces the data needed for the paper's experimental comparison table
    (showing when GenZ-ICP flags degeneracy but IV-GICP remains well-conditioned).

    Args:
        F_geo:              (6, 6) geometric FIM
        F_photo:            (6, 6) photometric FIM
        hessian_6x6:        (6, 6) GN Hessian (for GenZ-ICP metric); if None,
                            uses F_geo as Hessian proxy
        degeneracy_threshold:  λ_min threshold for IV-GICP/geometric FIM (default 1e-3)
        genz_kappa_threshold:  κ threshold for GenZ-ICP flag (default 100.0).
                               GenZ-ICP (Lee et al., RA-L 2025) uses κ > threshold
                               to detect ill-conditioned translational optimization.

    Returns dict with:
        iv_gicp_min_eig:      λ_min(F_combined)  — stays > 0 by Theorem 1
        iv_gicp_cond:         cond(F_combined)
        iv_gicp_degenerate:   λ_min(F_combined) < degeneracy_threshold
        geo_min_eig:          λ_min(F_geo)       — drops in corridor
        geo_degenerate:       True if geometry alone is degenerate
        genz_icp_kappa:       κ = √(λ_max/λ_min) of translational Hessian block
        genz_icp_degenerate:  True if κ > genz_kappa_threshold
        intensity_contribution: λ_min(F_combined) - λ_min(F_geo)  — intensity lift
        rescued_by_intensity:  geo_degenerate and NOT iv_gicp_degenerate
    """
    H = hessian_6x6 if hessian_6x6 is not None else F_geo
    F_combined = F_geo + F_photo

    eig_combined = np.linalg.eigvalsh(F_combined)
    eig_geo = np.linalg.eigvalsh(F_geo)

    min_combined = float(eig_combined[0])
    min_geo = float(eig_geo[0])
    cond_combined = float(eig_combined[-1] / max(min_combined, 1e-15))
    genz_kappa = genz_icp_condition_number(H)

    geo_deg = min_geo < degeneracy_threshold
    iv_deg = min_combined < degeneracy_threshold
    genz_deg = genz_kappa > genz_kappa_threshold

    return {
        "iv_gicp_min_eig": min_combined,
        "iv_gicp_cond": cond_combined,
        "iv_gicp_degenerate": iv_deg,
        "geo_min_eig": min_geo,
        "geo_degenerate": geo_deg,
        "genz_icp_kappa": genz_kappa,
        "genz_icp_degenerate": genz_deg,
        "intensity_contribution": min_combined - min_geo,
        "rescued_by_intensity": geo_deg and not iv_deg,
    }


class DegeneracyMonitor:
    """
    Runtime degeneracy monitor for IV-GICP pipeline.

    Records per-frame FIM analysis for generating paper figures
    showing where geometric degeneracy occurs and where intensity
    rescues it (e.g., tunnel/corridor sequences).
    """

    def __init__(self):
        self.frame_reports: List[DegeneracyReport] = []
        self.frame_ids: List[int] = []

    def record(self, frame_id: int, report: DegeneracyReport) -> None:
        self.frame_reports.append(report)
        self.frame_ids.append(frame_id)

    def get_condition_numbers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (geo_condition_numbers, combined_condition_numbers) arrays."""
        geo = np.array([r.geo_condition_number for r in self.frame_reports])
        combined = np.array([r.condition_number for r in self.frame_reports])
        return geo, combined

    def get_min_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (geo_min_eig, combined_min_eig) arrays."""
        geo = np.array([r.geometric_eigenvalues[0] for r in self.frame_reports])
        combined = np.array([r.combined_eigenvalues[0] for r in self.frame_reports])
        return geo, combined

    def get_rescue_frames(self) -> List[int]:
        """Return frame IDs where photometric rescue occurred."""
        return [
            fid for fid, r in zip(self.frame_ids, self.frame_reports)
            if r.is_geo_degenerate and r.is_rescued
        ]

    def summary(self) -> Dict:
        """Summary statistics for the paper."""
        if not self.frame_reports:
            return {}
        geo_min, combined_min = self.get_min_eigenvalues()
        geo_cond, combined_cond = self.get_condition_numbers()
        n_degenerate = sum(1 for r in self.frame_reports if r.is_geo_degenerate)
        n_rescued = sum(1 for r in self.frame_reports if r.is_rescued)
        return {
            "n_frames": len(self.frame_reports),
            "n_geo_degenerate": n_degenerate,
            "n_photometric_rescued": n_rescued,
            "rescue_rate": n_rescued / max(n_degenerate, 1),
            "geo_min_eigenvalue_mean": float(np.mean(geo_min)),
            "combined_min_eigenvalue_mean": float(np.mean(combined_min)),
            "geo_condition_number_mean": float(np.mean(geo_cond)),
            "combined_condition_number_mean": float(np.mean(combined_cond)),
        }
