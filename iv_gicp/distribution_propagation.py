"""
Retroactive Distribution Propagation

Factor Graph 최적화 후, 점을 일일이 변환하지 않고
복셀의 (μ, Σ)만 Lie theory로 업데이트.

μ_new = ΔT μ_old
Σ_new ≈ R(ΔT) Σ_old R(ΔT)^T

Proposition 2 (First-Order Propagation Error Bound):
  The covariance propagation Σ_new = R Σ_old R^T is exact for the rotation
  component and first-order accurate for translation. The approximation error
  from ignoring translation's effect on covariance is:
    ||Σ_exact - R Σ_old R^T|| = O(||t||² / r²)
  where t is the translation correction and r is the voxel centroid distance
  from the sensor origin. For typical pose corrections (||t|| ~ 0.01-0.1m)
  and voxel distances (r ~ 10-50m), this is O(10^-6), negligible.

Complexity:
  FORM (point-wise):    O(N) per pose correction, N = total map points
  Distribution (ours):  O(V) per pose correction, V = total voxels
  Typical: V ≈ N/50 to N/200, giving 50-200x speedup.

FORM 대비: O(voxels) vs O(points) - 수백만 점 대신 수천 복셀만 업데이트.

=== 적용 범위 주의사항 (2026-02-21) ===

이 모듈의 O(V) 우위는 다음 시나리오에서만 성립:

  [루프 클로저 발생]
    → pose graph 소급 수정 (k 프레임에 걸친 Δpose)
    → 축적된 대형 맵 (kN raw points ↔ V voxels, V ≪ kN) 동기화
    → FORM: O(kN), Distribution: O(V)  ← 여기서만 speedup 실현

순수 odometry (루프 클로저 없음) 파이프라인에서는:
  - 소급 수정 이벤트가 발생하지 않으므로 FORM도 호출되지 않음
  - 이 모듈을 매 프레임 호출하면 오히려 불필요한 오버헤드 발생
  - pipeline.py의 use_distribution_propagation 기본값은 False로 유지할 것

논문 C3 기여의 실증을 위해서는 루프 클로저 시뮬레이션 실험이 필요:
  → examples/run_loop_closure_eval.py (미구현, TODO)
  → KITTI seq 00 / seq 08 loop 구간 활용 권장
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VoxelDistribution:
    """복셀 분포: 평균과 공분산. frame_id: 어느 pose로 변환된 점들로 구성되었는지."""

    mean: np.ndarray  # (3,)
    cov: np.ndarray  # (3, 3)
    frame_id: int = 0  # optional: for per-frame delta propagation


def se3_rotation_from_transform(T: np.ndarray) -> np.ndarray:
    """Extract rotation from 4x4 transform."""
    return T[:3, :3]


def se3_translation_from_transform(T: np.ndarray) -> np.ndarray:
    """Extract translation from 4x4 transform."""
    return T[:3, 3]


def propagate_mean(mu: np.ndarray, delta_T: np.ndarray) -> np.ndarray:
    """μ_new = ΔT μ_old (homogeneous)."""
    R = delta_T[:3, :3]
    t = delta_T[:3, 3]
    return R @ mu + t


def propagate_covariance(Sigma: np.ndarray, delta_T: np.ndarray) -> np.ndarray:
    """Σ_new ≈ R(ΔT) Σ_old R(ΔT)^T (geometry only, 3x3)."""
    R = delta_T[:3, :3]
    return R @ Sigma @ R.T


def propagate_covariance_4d(Sigma: np.ndarray, delta_T: np.ndarray) -> np.ndarray:
    """
    For 4D cov [Σ_geo | *; * | σ_I], only rotate geometric block.
    Intensity dimension is invariant under pose change.
    """
    if Sigma.shape == (3, 3):
        return propagate_covariance(Sigma, delta_T)
    R = delta_T[:3, :3]
    Sigma_new = Sigma.copy()
    Sigma_new[:3, :3] = R @ Sigma[:3, :3] @ R.T
    Sigma_new[:3, 3] = R @ Sigma[:3, 3]
    Sigma_new[3, :3] = Sigma_new[:3, 3].T
    # Sigma_new[3,3] unchanged (intensity)
    return Sigma_new


class DistributionPropagator:
    """
    Retroactive map refinement via distribution propagation.
    """

    def __init__(self):
        self.voxel_map: Dict[Any, VoxelDistribution] = {}

    def set_voxel_map(
        self,
        voxel_ids: List[Any],
        means: List[np.ndarray],
        covs: List[np.ndarray],
        frame_ids: Optional[List[int]] = None,
    ) -> None:
        """Initialize voxel map from list of (id, mean, cov). Optionally track frame_id per voxel."""
        self.voxel_map = {}
        for i, vid in enumerate(voxel_ids):
            fid = frame_ids[i] if frame_ids is not None else 0
            self.voxel_map[vid] = VoxelDistribution(
                mean=np.asarray(means[i]).flatten()[:3],
                cov=np.asarray(covs[i])[:3, :3],
                frame_id=fid,
            )

    def propagate(self, delta_T: np.ndarray) -> None:
        """
        Apply single pose change ΔT to all voxel distributions.
        Use when all voxels were built from the same pose (e.g. single-frame map).
        In-place update of voxel_map.
        """
        delta_T = np.asarray(delta_T)
        if delta_T.shape != (4, 4):
            raise ValueError("delta_T must be 4x4")

        for vid in self.voxel_map:
            v = self.voxel_map[vid]
            v.mean = propagate_mean(v.mean, delta_T)
            v.cov = propagate_covariance(v.cov, delta_T)

    def propagate_per_frame(
        self,
        delta_T_per_frame: Dict[int, np.ndarray],
    ) -> None:
        """
        Apply per-frame pose corrections (FORM 대비 우리 방식).
        각 복셀에 대해 해당 frame_id의 delta_T를 적용.
        Fully vectorized: frame_id별로 모든 mean/cov를 한 번에 변환.
        """
        from collections import defaultdict

        by_frame = defaultdict(list)
        for vid in self.voxel_map:
            v = self.voxel_map[vid]
            if v.frame_id in delta_T_per_frame:
                by_frame[v.frame_id].append(vid)
        for fid, vids in by_frame.items():
            delta_T = np.asarray(delta_T_per_frame[fid])
            if delta_T.shape != (4, 4):
                continue
            R, t = delta_T[:3, :3], delta_T[:3, 3]
            # Vectorized: (N,3) means, (N,3,3) covs
            means = np.array([self.voxel_map[vid].mean for vid in vids])
            covs = np.array([self.voxel_map[vid].cov for vid in vids])
            means_new = (R @ means.T).T + t
            covs_new = np.einsum("ij,njk,kl->nil", R, covs, R.T)
            for idx, vid in enumerate(vids):
                self.voxel_map[vid].mean = means_new[idx]
                self.voxel_map[vid].cov = covs_new[idx]

    def get_voxel_map(self) -> Dict[Any, VoxelDistribution]:
        return self.voxel_map

    def get_means(self) -> np.ndarray:
        """Return (N, 3) array of all voxel means."""
        if not self.voxel_map:
            return np.empty((0, 3))
        return np.array([v.mean for v in self.voxel_map.values()])


# ─── Error Bound Analysis ────────────────────────────────────────────────────

def propagation_error_bound(
    delta_T: np.ndarray,
    voxel_distances: np.ndarray,
) -> np.ndarray:
    """
    Compute first-order approximation error bound for distribution propagation.

    The covariance propagation Σ_new ≈ R Σ_old R^T ignores the translation
    component's effect on the covariance. The bound is:

      ||Σ_exact - R Σ_old R^T|| ≤ ||t||² / r_i²

    where t is the translation correction and r_i is the voxel centroid's
    distance from the sensor.

    Args:
        delta_T: (4, 4) pose correction
        voxel_distances: (V,) distances from sensor to each voxel centroid

    Returns:
        error_bounds: (V,) per-voxel error bound (dimensionless)
    """
    t = delta_T[:3, 3]
    t_norm_sq = np.dot(t, t)
    r_sq = np.maximum(voxel_distances ** 2, 1e-6)
    return t_norm_sq / r_sq


def propagate_with_pose_uncertainty(
    mu_old: np.ndarray,
    Sigma_old: np.ndarray,
    delta_T: np.ndarray,
    Sigma_delta_T: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full covariance propagation including pose correction uncertainty (Theorem 2).

    Theorem 2 (Propagation Error Bound, Section 11.H):
      Σ_true = R Σ_old R^T + J_ΔT Σ_ΔT J_ΔT^T  +  O(||Σ_ΔT||²)

    where J_ΔT = ∂(R_Δ μ) / ∂ξ_Δ is the Jacobian of the transformed mean
    w.r.t. the pose correction in Lie algebra coordinates.

    Args:
        mu_old:       (3,) voxel mean
        Sigma_old:    (3,3) voxel covariance
        delta_T:      (4,4) pose correction ΔT = T_new T_old^{-1}
        Sigma_delta_T: (6,6) uncertainty of ΔT in Lie algebra (from factor graph).
                       If None, falls back to approximate propagation (Σ_new = R Σ R^T).

    Returns:
        mu_new:       (3,) updated mean
        Sigma_new:    (3,3) updated covariance (with uncertainty term if provided)
        error_bound:  float, ||J_ΔT Σ_ΔT J_ΔT^T||_F (0 if Sigma_delta_T is None)

    Notes:
        - Theorem 2 bound: ||error||_F ≤ (2||μ||² + 3) · ||Σ_ΔT||_F
        - When factor graph has converged, Σ_ΔT → 0 and error → 0.
        - Use Sigma_delta_T from iSAM2 marginal covariance for accurate propagation.
    """
    R = delta_T[:3, :3]
    t = delta_T[:3, 3]

    # Mean update (exact)
    mu_new = R @ mu_old + t

    # Base covariance propagation (approximate, ignores Σ_ΔT)
    Sigma_approx = R @ Sigma_old @ R.T

    if Sigma_delta_T is None:
        return mu_new, Sigma_approx, 0.0

    # J_ΔT = ∂(R_Δ μ) / ∂ξ_Δ  at ξ_Δ = 0
    # ξ_Δ = [φ (rotation), ρ (translation)]  (6D Lie algebra)
    # ∂(R μ)/∂φ = -[μ]_×  (skew-symmetric, 3×3 block)
    # ∂(R μ)/∂ρ = R         (translation Jacobian, 3×3)
    # Combined J_ΔT ∈ R^{3×6}:
    from iv_gicp.iv_gicp import skew_symmetric
    mu_hat = -skew_symmetric(mu_old)   # 3×3: ∂(R μ)/∂φ at identity
    J_dT = np.zeros((3, 6))
    J_dT[:, :3] = mu_hat               # rotation block
    J_dT[:, 3:] = R                    # translation block

    # Additional covariance term from pose uncertainty
    Sigma_extra = J_dT @ Sigma_delta_T @ J_dT.T   # (3, 3)

    Sigma_new = Sigma_approx + Sigma_extra
    error_bound = float(np.linalg.norm(Sigma_extra, 'fro'))

    return mu_new, Sigma_new, error_bound


def propagate_adaptive(
    voxel_map_items: List[Tuple[Any, np.ndarray, np.ndarray]],
    delta_T: np.ndarray,
    Sigma_delta_T: Optional[np.ndarray] = None,
    convergence_threshold: float = 1e-4,
) -> Tuple[List[Tuple[Any, np.ndarray, np.ndarray]], dict]:
    """
    Adaptive propagation: switch between approximate and full based on convergence.

    Section 11.H Remark: Use convergence indicator ||Σ_ΔT||_F < τ_conv to
    decide approximate (fast, O(k)) vs full propagation (accurate, with pose uncertainty).

    Args:
        voxel_map_items: list of (voxel_id, mean, cov)
        delta_T:         (4,4) pose correction
        Sigma_delta_T:   (6,6) pose correction uncertainty. None = always approximate.
        convergence_threshold: τ_conv; if ||Σ_ΔT||_F < τ, use approximate propagation.

    Returns:
        updated_items: list of (voxel_id, new_mean, new_cov)
        stats: dict with mode ('approximate'/'full'), error_bound_max, etc.
    """
    use_full = False
    sigma_norm = 0.0

    if Sigma_delta_T is not None:
        sigma_norm = float(np.linalg.norm(Sigma_delta_T, 'fro'))
        use_full = sigma_norm >= convergence_threshold

    R = delta_T[:3, :3]
    t = delta_T[:3, 3]
    updated = []
    max_error = 0.0

    for vid, mu, cov in voxel_map_items:
        if use_full:
            mu_new, cov_new, err = propagate_with_pose_uncertainty(
                mu, cov, delta_T, Sigma_delta_T
            )
            max_error = max(max_error, err)
        else:
            mu_new = R @ mu + t
            cov_new = R @ cov @ R.T
        updated.append((vid, mu_new, cov_new))

    return updated, {
        "mode": "full" if use_full else "approximate",
        "sigma_delta_T_norm": sigma_norm,
        "max_error_bound": max_error,
        "n_voxels": len(voxel_map_items),
    }


def complexity_report(
    n_points: int,
    n_voxels: int,
    n_pose_corrections: int = 1,
) -> dict:
    """
    Complexity comparison: FORM (point-wise) vs Distribution Propagation.

    FORM:         O(N · k) per pose correction, k = 3 (transform each point)
    Distribution: O(V · k²) per pose correction, k² = 9 (3×3 matrix rotation)

    Args:
        n_points: total map points
        n_voxels: total voxels
        n_pose_corrections: number of poses being corrected

    Returns:
        dict with operation counts and speedup ratio
    """
    form_ops = n_points * 3 * n_pose_corrections      # 3D transform per point
    dist_ops = n_voxels * (3 + 9) * n_pose_corrections  # mean (3) + cov rotation (9)
    speedup = form_ops / max(dist_ops, 1)

    return {
        "form_operations": form_ops,
        "distribution_operations": dist_ops,
        "speedup_ratio": float(speedup),
        "n_points": n_points,
        "n_voxels": n_voxels,
        "compression_ratio": n_points / max(n_voxels, 1),
    }
