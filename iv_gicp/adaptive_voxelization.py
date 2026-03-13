"""
Adaptive Voxelization: Information-theoretic octree-based voxelization.

Splitting criteria:
  1. Shannon entropy: H(X) = (1/2) ln |(2πe)^k Σ|
  2. Information Gain (KL-divergence): IG = H(parent) - Σ(n_k/n) H(child_k)
  3. Intensity variance: captures photometric heterogeneity (lane markings, textures)

The information gain formulation provides principled justification for adaptive
splitting: subdivide when a single Gaussian poorly approximates the underlying
point distribution (high IG) or when photometric texture is present (high var_I).
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class VoxelStats:
    """Voxel statistics: mean, covariance, point count, intensity variance."""

    mean: np.ndarray  # (3,) geometric mean
    cov: np.ndarray  # (3,3) geometric covariance
    mean_intensity: float
    var_intensity: float
    n_points: int
    points: Optional[np.ndarray] = None  # (N, 4) [x,y,z,I] for leaf
    point_covs: Optional[np.ndarray] = None  # (N, 3, 3) per-point covariances


def compute_shannon_entropy(cov: np.ndarray, k: int = 3) -> float:
    """
    Shannon differential entropy for multivariate Gaussian:
      H(X) = (1/2) ln |(2πe)^k Σ|

    This measures the "spread" or uncertainty of the point distribution.
    High entropy → complex/scattered structure → should subdivide.
    Low entropy → well-approximated by single Gaussian → keep as single voxel.
    """
    det = np.linalg.det(cov + 1e-8 * np.eye(cov.shape[0]))
    return 0.5 * np.log((2 * np.pi * np.e) ** k * np.maximum(det, 1e-20))


def compute_voxel_stats(
    points: np.ndarray,
    intensities: np.ndarray,
    point_covs: Optional[np.ndarray] = None,
) -> VoxelStats:
    """
    Compute mean, covariance, and intensity variance for a set of points.
    points: (N, 3) xyz coordinates
    intensities: (N,) intensity values
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)
    xyz = points[:, :3]

    mean = np.mean(xyz, axis=0)
    if len(xyz) > 1:
        centered = xyz - mean
        # Use direct BLAS matmul instead of np.cov to avoid Python overhead.
        # Equivalent: np.cov(centered.T) = (centered.T @ centered) / (n-1)
        n = len(xyz)
        cov = (centered.T @ centered) / (n - 1)
        cov = np.atleast_2d(cov)
        if cov.shape != (3, 3):
            cov = np.eye(3) * 1e-6
    else:
        cov = np.eye(3) * 1e-6

    mean_i = float(np.mean(intensities))
    var_i = float(np.var(intensities)) if len(intensities) > 1 else 0.0

    return VoxelStats(
        mean=mean,
        cov=cov,
        mean_intensity=mean_i,
        var_intensity=var_i,
        n_points=len(xyz),
        points=np.column_stack([xyz, intensities]) if len(xyz) <= 200 else None,
        point_covs=point_covs,
    )


def update_stats_incremental(
    stats: VoxelStats,
    new_xyz: np.ndarray,
    new_intensity: np.ndarray,
) -> VoxelStats:
    """
    Welford's online algorithm for incremental mean/covariance update.
    Avoids full rebuild when adding a small batch of points.

    For batch of m new points added to existing n-point distribution:
      mean_new = (n * mean_old + sum(new_xyz)) / (n + m)
      cov_new via online update rule

    Args:
        stats: existing voxel statistics
        new_xyz: (m, 3) new point coordinates
        new_intensity: (m,) new intensity values
    """
    n_old = stats.n_points
    m = len(new_xyz)
    n_new = n_old + m

    if n_new < 2:
        return stats

    # Mean update
    sum_old = stats.mean * n_old
    sum_new_xyz = np.sum(new_xyz, axis=0)
    mean_new = (sum_old + sum_new_xyz) / n_new

    # Covariance update (parallel algorithm for combining two sample sets)
    # Ref: Chan, Golub, LeVeque (1979)
    delta = np.mean(new_xyz, axis=0) - stats.mean
    if m > 1:
        cov_batch = np.cov(new_xyz.T)
        cov_batch = np.atleast_2d(cov_batch)
        if cov_batch.shape != (3, 3):
            cov_batch = np.eye(3) * 1e-6
    else:
        cov_batch = np.eye(3) * 1e-6

    cov_new = ((n_old - 1) * stats.cov + (m - 1) * cov_batch + (n_old * m / n_new) * np.outer(delta, delta)) / (
        n_new - 1
    )

    # Intensity update
    sum_i_old = stats.mean_intensity * n_old
    sum_i_new = float(np.sum(new_intensity))
    mean_i_new = (sum_i_old + sum_i_new) / n_new

    # Intensity variance (Welford for variance)
    delta_i = np.mean(new_intensity) - stats.mean_intensity
    if m > 1:
        var_batch = float(np.var(new_intensity))
    else:
        var_batch = 0.0
    var_i_new = ((n_old - 1) * stats.var_intensity + (m - 1) * var_batch + (n_old * m / n_new) * delta_i**2) / max(
        n_new - 1, 1
    )

    return VoxelStats(
        mean=mean_new,
        cov=cov_new,
        mean_intensity=mean_i_new,
        var_intensity=var_i_new,
        n_points=n_new,
    )


class OctreeNode:
    """Octree node for adaptive voxelization."""

    def __init__(self, center: np.ndarray, half_size: float, stats: Optional[VoxelStats] = None):
        self.center = np.asarray(center, dtype=float)
        self.half_size = half_size
        self.stats = stats
        self.children: List[OctreeNode] = []
        self.is_leaf = True

    def get_voxel_key(self) -> Tuple[int, int, int]:
        """Integer voxel indices for hashing."""
        return tuple((self.center / (2 * self.half_size + 1e-9)).astype(int))

    def _estimate_information_gain(
        self,
        xyz: np.ndarray,
        intensities: np.ndarray,
        min_points: int,
    ) -> float:
        """
        Estimate information gain from splitting this node into 8 octants.

        IG = H(parent) - Σ (n_k / n) · H(child_k)

        This approximates the KL-divergence between the single-Gaussian model
        and the Gaussian mixture model of children. High IG means the parent
        distribution is poorly approximated by a single Gaussian.

        Proposition (Information Gain Splitting):
          Splitting is justified when IG > τ, i.e., the single Gaussian loses
          more than τ nats of information compared to the mixture model.
          This is equivalent to: KL(GMM || single) > τ for the special case
          of uniform-weight octant children.
        """
        H_parent = compute_shannon_entropy(self.stats.cov)
        n = len(xyz)
        if n < min_points * 2:
            return 0.0

        # Trial-split into octants
        hs = self.half_size / 2
        offsets = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        child_centers = self.center + offsets * hs

        # Vectorized: compute all 8 octant masks at once (no Python loop)
        # xyz (N, 3), child_centers (8, 3) → diffs (8, N, 3)
        diffs = np.abs(xyz[np.newaxis, :, :] - child_centers[:, np.newaxis, :])  # (8, N, 3)
        masks = np.all(diffs <= hs + 1e-6, axis=2)  # (8, N) bool

        weighted_H_children = 0.0
        n_valid = 0
        for k in range(8):
            n_k = int(np.sum(masks[k]))
            if n_k < min_points:
                continue
            pts_k = xyz[masks[k]]
            if n_k > 1:
                mean_k = np.mean(pts_k, axis=0)
                centered_k = pts_k - mean_k
                cov_k = (centered_k.T @ centered_k) / (n_k - 1)
                cov_k = np.atleast_2d(cov_k)
                if cov_k.shape != (3, 3):
                    cov_k = np.eye(3) * 1e-6
            else:
                cov_k = np.eye(3) * 1e-6
            H_k = compute_shannon_entropy(cov_k)
            weighted_H_children += (n_k / n) * H_k
            n_valid += n_k

        if n_valid < min_points:
            return 0.0

        return H_parent - weighted_H_children

    def subdivide(
        self,
        xyz: np.ndarray,
        intensities: np.ndarray,
        point_covs: Optional[np.ndarray],
        entropy_threshold: float,
        intensity_var_threshold: float,
        min_points: int,
        max_depth: int,
        depth: int = 0,
    ) -> None:
        """
        Recursively subdivide based on information gain and intensity variance.

        Splitting criterion (information-theoretic):
          1. IG > entropy_threshold:  geometric information gain justifies split
          2. H > H_abs_max:  raw entropy exceeds absolute cap (always split large voxels)
          3. var_I > intensity_var_threshold:  photometric heterogeneity
        """
        if depth >= max_depth or self.stats.n_points < min_points:
            return

        H_parent = compute_shannon_entropy(self.stats.cov)
        IG = self._estimate_information_gain(xyz, intensities, min_points)

        # Dual criterion: IG for information-theoretic justification,
        # raw entropy as fallback for clearly complex distributions
        H_abs_max = entropy_threshold + 5.0  # absolute entropy cap
        should_split = (
            IG > entropy_threshold or H_parent > H_abs_max or self.stats.var_intensity > intensity_var_threshold
        )

        if not should_split:
            return

        # Split into 8 children
        hs = self.half_size / 2
        offsets = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ]
        )
        child_centers = self.center + offsets * hs

        # Vectorized: compute all 8 split masks at once
        diffs_split = np.abs(xyz[np.newaxis, :, :] - child_centers[:, np.newaxis, :])  # (8, N, 3)
        masks_split = np.all(diffs_split <= hs + 1e-6, axis=2)  # (8, N) bool

        for k, cc in enumerate(child_centers):
            mask = masks_split[k]
            if not np.any(mask):
                continue
            xyz_c = xyz[mask]
            int_c = intensities[mask]
            cov_c = point_covs[mask] if point_covs is not None else None
            child_stats = compute_voxel_stats(xyz_c, int_c, cov_c)
            child = OctreeNode(cc, hs, child_stats)
            child.subdivide(
                xyz_c,
                int_c,
                cov_c,
                entropy_threshold,
                intensity_var_threshold,
                min_points,
                max_depth,
                depth + 1,
            )
            self.children.append(child)

        if self.children:
            self.is_leaf = False
            self.stats = None  # Internal node: no direct stats


class AdaptiveVoxelMap:
    """
    Adaptive voxel map using information-theoretic octree splitting.

    Supports two modes:
      1. Full rebuild: build() - O(N log N) octree construction
      2. Incremental: insert_points() - O(m) per batch via Welford update
    """

    def __init__(
        self,
        voxel_size: float = 1.0,
        entropy_threshold: float = 2.0,
        intensity_var_threshold: float = 100.0,
        min_points_per_voxel: int = 3,
        max_depth: int = 8,
    ):
        self.voxel_size = voxel_size
        self.entropy_threshold = entropy_threshold
        self.intensity_var_threshold = intensity_var_threshold
        self.min_points = min_points_per_voxel
        self.max_depth = max_depth
        self.leaves: List[OctreeNode] = []
        self.voxel_dict: dict = {}  # key -> VoxelStats
        self._root: Optional[OctreeNode] = None

    def build(
        self, points: np.ndarray, intensities: Optional[np.ndarray] = None, point_covs: Optional[np.ndarray] = None
    ) -> None:
        """
        Build adaptive voxel map from point cloud (full rebuild).
        points: (N, 3) or (N, 4) [x,y,z,I]
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
        xyz = points[:, :3]
        if intensities is None:
            intensities = points[:, 3] if points.shape[1] >= 4 else np.zeros(len(points))

        # Reset
        self.leaves = []
        self.voxel_dict = {}

        # Initial root covering whole cloud
        center = np.mean(xyz, axis=0)
        half_size = max(np.ptp(xyz, axis=0)) / 2 + self.voxel_size
        root_stats = compute_voxel_stats(xyz, intensities, point_covs)
        self._root = OctreeNode(center, half_size, root_stats)

        self._root.subdivide(
            xyz,
            intensities,
            point_covs,
            self.entropy_threshold,
            self.intensity_var_threshold,
            self.min_points,
            self.max_depth,
        )

        self._collect_leaves(self._root)
        self._build_voxel_dict()

    def insert_points(
        self,
        new_xyz: np.ndarray,
        new_intensities: np.ndarray,
    ) -> None:
        """
        Incrementally update voxel statistics without full octree rebuild.
        Uses Welford's online algorithm for O(m) update per batch.

        Points are assigned to existing leaf nodes based on spatial proximity.
        Leaves that exceed splitting thresholds after update are NOT split
        (deferred to next full rebuild for simplicity).
        """
        if not self.leaves:
            return

        # Build leaf center array for fast assignment
        leaf_centers = np.array([l.center for l in self.leaves])
        leaf_half_sizes = np.array([l.half_size for l in self.leaves])

        for i in range(len(new_xyz)):
            pt = new_xyz[i]
            # Find containing leaf (nearest center within bounds)
            dists = np.abs(leaf_centers - pt)
            in_bounds = np.all(dists <= leaf_half_sizes[:, np.newaxis] + 1e-6, axis=1)

            if not np.any(in_bounds):
                continue

            # Pick the smallest containing leaf
            candidates = np.where(in_bounds)[0]
            best = candidates[np.argmin(leaf_half_sizes[candidates])]
            leaf = self.leaves[best]

            if leaf.stats is not None:
                leaf.stats = update_stats_incremental(
                    leaf.stats,
                    pt.reshape(1, 3),
                    np.array([new_intensities[i]]),
                )

    def _collect_leaves(self, node: OctreeNode) -> None:
        if node.is_leaf and node.stats is not None:
            self.leaves.append(node)
        for c in node.children:
            self._collect_leaves(c)

    def _build_voxel_dict(self) -> None:
        self.voxel_dict = {}
        for i, leaf in enumerate(self.leaves):
            key = leaf.get_voxel_key()
            self.voxel_dict[key] = leaf.stats

    def get_voxels(self) -> List[Tuple[np.ndarray, np.ndarray, float, int]]:
        """Returns list of (mean, cov, mean_intensity, n_points) for each voxel."""
        out = []
        for leaf in self.leaves:
            if leaf.stats is not None:
                out.append(
                    (
                        leaf.stats.mean,
                        leaf.stats.cov,
                        leaf.stats.mean_intensity,
                        leaf.stats.n_points,
                    )
                )
        return out

    def get_voxel_count(self) -> int:
        return len(self.leaves)

    def compute_fim_contribution(
        self,
        voxel_size: float,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """
        Estimate FIM contribution trace per voxel (Section 11.G).

        Implements the Information-Optimal Voxelization claim:
          tr(I_v) ∝ n_v · tr(Σ_v^{-1})    (geometric component)
                  + n_v · ω_I_v             (intensity component)

        where ω_I_v = α² / (Var(I)/ℓ_v² + ε) is the intensity precision proxy.

        Returns:
            contributions: (N_voxels, 2) array of [tr_geo, tr_int] per voxel,
                           sorted by total contribution (descending).
        """
        from iv_gicp.iv_gicp import build_photometric_sigma_sq

        contributions = []
        for leaf in self.leaves:
            if leaf.stats is None:
                continue
            s = leaf.stats
            n = s.n_points

            # Geometric FIM contribution: n · tr(Σ^{-1})
            try:
                cov_reg = s.cov + 1e-6 * np.eye(3)
                omega_geo = np.linalg.inv(cov_reg)
                tr_geo = float(n * np.trace(omega_geo))
            except np.linalg.LinAlgError:
                tr_geo = 0.0

            # Intensity FIM contribution: n · (1/σ_I²)
            sigma_sq = build_photometric_sigma_sq(s.var_intensity, voxel_size, alpha)
            tr_int = float(n / (sigma_sq + 1e-12))

            contributions.append([tr_geo, tr_int, n])

        if not contributions:
            return np.zeros((0, 3))

        arr = np.array(contributions)  # (V, 3): [tr_geo, tr_int, n_pts]
        # Sort by total FIM contribution descending
        total = arr[:, 0] + arr[:, 1]
        arr = arr[np.argsort(-total)]
        return arr

    def fim_summary(self, voxel_size: float, alpha: float = 0.1) -> dict:
        """
        FIM-based voxel map quality summary (Section 11.E unified framework).

        Returns dict with:
          total_tr_geo:   Σ_v tr(I_v^geo)  -- maximized by C1
          total_tr_int:   Σ_v tr(I_v^int)  -- filled by C2
          n_low_fim:      voxels with low total FIM (poor information content)
          top_k_voxels:   fraction of FIM in top 10% of voxels
        """
        contribs = self.compute_fim_contribution(voxel_size, alpha)
        if len(contribs) == 0:
            return {}

        tr_geo = float(np.sum(contribs[:, 0]))
        tr_int = float(np.sum(contribs[:, 1]))
        total_per_voxel = contribs[:, 0] + contribs[:, 1]
        total = float(np.sum(total_per_voxel))

        k10 = max(1, len(contribs) // 10)
        top10_fraction = float(np.sum(total_per_voxel[:k10]) / max(total, 1e-12))

        return {
            "total_tr_geo": tr_geo,
            "total_tr_int": tr_int,
            "total_tr_fim": tr_geo + tr_int,
            "intensity_pct": 100.0 * tr_int / max(total, 1e-12),
            "n_voxels": len(contribs),
            "n_low_fim": int(np.sum(total_per_voxel < np.percentile(total_per_voxel, 10))),
            "top10_pct_of_fim": 100.0 * top10_fraction,
        }
