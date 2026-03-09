"""
Factor Graph Optimization for Fixed-Lag Odometry (FORM-style)

FORM: Fixed-Lag Odometry with Reparative Mapping
- Densely connected factor graph (odometry factors between consecutive poses)
- Fixed-lag smoothing: sliding window of last K poses
- Output: corrected poses for map update
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .se3_utils import (
    se3_exp,
    se3_log,
    se3_inverse,
    se3_compose,
    se3_to_matrix,
    adjoint_se3,
)


@dataclass
class OdometryFactor:
    """Relative pose constraint between frame i and j."""

    i: int
    j: int
    T_meas: np.ndarray  # measured relative pose T_j^{-1} T_i (pose j from i)
    omega: np.ndarray = None  # 6x6 information matrix (default: identity)

    def __post_init__(self):
        if self.omega is None:
            self.omega = np.eye(6)


@dataclass
class PriorFactor:
    """Absolute prior on pose 0."""

    T_prior: np.ndarray  # prior pose
    omega: np.ndarray = None  # 6x6 information matrix

    def __post_init__(self):
        if self.omega is None:
            self.omega = np.eye(6)


class PoseGraphOptimizer:
    """
    Fixed-lag pose graph optimizer.
    Optimizes poses in a sliding window using Gauss-Newton.
    """

    def __init__(
        self,
        lag_size: int = 10,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-6,
        odometry_weight: float = 1.0,
        prior_weight: float = 100.0,
    ):
        self.lag_size = lag_size
        self.max_iter = max_iterations
        self.conv_thresh = convergence_threshold
        self.odometry_weight = odometry_weight
        self.prior_weight = prior_weight

        self.poses: List[np.ndarray] = []
        self.odom_factors: List[OdometryFactor] = []
        self.prior: Optional[PriorFactor] = None

    def add_pose(self, T: np.ndarray) -> int:
        """Add a new pose. Returns frame index."""
        idx = len(self.poses)
        self.poses.append(np.array(T, dtype=float))
        return idx

    def add_odometry_factor(
        self, i: int, j: int, T_rel: np.ndarray,
        omega: Optional[np.ndarray] = None,
    ) -> None:
        """Add odometry factor: T_rel is pose of j in frame i (T_i^{-1} @ T_j).

        omega: 6×6 information matrix (default: identity × odometry_weight).
               Pass the ICP Hessian for FORM-style information-weighted smoothing.
        """
        if omega is None:
            omega = np.eye(6) * self.odometry_weight
        self.odom_factors.append(OdometryFactor(i=i, j=j, T_meas=T_rel, omega=omega))

    def set_prior(self, T: np.ndarray, omega: Optional[np.ndarray] = None) -> None:
        """Set prior on first pose. omega overrides default prior_weight * I."""
        if omega is None:
            omega = np.eye(6) * self.prior_weight
        self.prior = PriorFactor(T_prior=T, omega=omega)

    def _get_factors_in_window(self, window_end: int) -> Tuple[List, List]:
        """Get factors affecting poses in the sliding window."""
        window_start = max(0, window_end - self.lag_size + 1)
        odom_in = []
        for f in self.odom_factors:
            if f.i >= window_start and f.j <= window_end:
                odom_in.append(f)
        return odom_in, window_start

    def _residual_odom(self, poses: List[np.ndarray], f: OdometryFactor) -> np.ndarray:
        """Residual for odometry factor: log(T_i^{-1} T_j T_meas^{-1})."""
        T_i = poses[f.i]
        T_j = poses[f.j]
        T_pred = se3_compose(se3_inverse(T_i), T_j)
        T_res = se3_compose(T_pred, se3_inverse(f.T_meas))
        return se3_log(T_res)

    def _residual_prior(self, poses: List[np.ndarray]) -> np.ndarray:
        """Residual for prior: log(T_0^{-1} T_prior)."""
        T_res = se3_compose(se3_inverse(poses[0]), self.prior.T_prior)
        return se3_log(T_res)

    def _build_system(
        self,
        poses: List[np.ndarray],
        window_start: int,
        window_end: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build H and b for Gauss-Newton. State: delta in Lie algebra for each pose."""
        n_poses = window_end - window_start + 1
        H = np.zeros((6 * n_poses, 6 * n_poses))
        b = np.zeros(6 * n_poses)

        def idx(k: int) -> int:
            return 6 * (k - window_start)

        # Odometry factors
        odom_in, _ = self._get_factors_in_window(window_end)
        for f in odom_in:
            r = self._residual_odom(poses, f)
            sqrt_omega = np.sqrt(np.diag(np.diag(f.omega)))
            r_w = sqrt_omega @ r

            # Jacobians: r w.r.t. pose_i and pose_j
            # r = log(T_i^{-1} T_j T_meas^{-1})
            # dr/d(delta_i) = -Ad(T_i^{-1} T_j T_meas^{-1})^{-1} @ Ad(T_i^{-1})  (left)
            # dr/d(delta_j) = Ad(T_j^{-1})  (right)
            T_i, T_j = poses[f.i], poses[f.j]
            T_pred = se3_compose(se3_inverse(T_i), T_j)
            T_res = se3_compose(T_pred, se3_inverse(f.T_meas))
            Ad_res_inv = np.linalg.inv(adjoint_se3(T_res))
            Ad_Ti_inv = adjoint_se3(se3_inverse(T_i))
            J_i = -Ad_res_inv @ Ad_Ti_inv
            J_j = Ad_res_inv @ adjoint_se3(se3_inverse(T_j))

            J_w = sqrt_omega @ np.hstack([J_i, J_j])
            ii, jj = idx(f.i), idx(f.j)
            H[ii : ii + 6, ii : ii + 6] += J_i.T @ f.omega @ J_i
            H[ii : ii + 6, jj : jj + 6] += J_i.T @ f.omega @ J_j
            H[jj : jj + 6, ii : ii + 6] += J_j.T @ f.omega @ J_i
            H[jj : jj + 6, jj : jj + 6] += J_j.T @ f.omega @ J_j
            b[ii : ii + 6] -= J_i.T @ f.omega @ r
            b[jj : jj + 6] -= J_j.T @ f.omega @ r

        # Prior factor
        if self.prior is not None and window_start == 0:
            r = self._residual_prior(poses)
            J = -np.linalg.inv(adjoint_se3(se3_compose(se3_inverse(poses[0]), self.prior.T_prior)))
            ii = idx(0)
            H[ii : ii + 6, ii : ii + 6] += J.T @ self.prior.omega @ J
            b[ii : ii + 6] -= J.T @ self.prior.omega @ r

        return H, b

    def optimize(self, window_end: Optional[int] = None) -> List[np.ndarray]:
        """
        Run Gauss-Newton optimization on the sliding window.
        Returns optimized poses (same length as input).
        """
        if not self.poses:
            return []
        window_end = window_end if window_end is not None else len(self.poses) - 1
        window_start = max(0, window_end - self.lag_size + 1)
        poses = [p.copy() for p in self.poses]

        for _ in range(self.max_iter):
            H, b = self._build_system(poses, window_start, window_end)
            n = 6 * (window_end - window_start + 1)
            H = H[:n, :n] + 1e-8 * np.eye(n)
            b = b[:n]
            try:
                delta = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break

            for k in range(window_start, window_end + 1):
                i = 6 * (k - window_start)
                xi = delta[i : i + 6]
                R, t = se3_exp(xi)
                T_delta = se3_to_matrix(R, t)
                poses[k] = se3_compose(poses[k], T_delta)

            if np.linalg.norm(delta) < self.conv_thresh:
                break

        return poses


class FormMapUpdater:
    """
    FORM-style map update: point-wise transformation when poses change.

    When pose T_old -> T_new for frame i:
    p_new = T_new @ T_old^{-1} @ p_old

    O(n) per pose change (n = number of points in that frame).
    """

    def __init__(self):
        self.points: np.ndarray = np.empty((0, 4))  # [x,y,z,I]
        self.frame_ids: np.ndarray = np.empty(0, dtype=int)  # which frame each point belongs to
        self.poses_old: List[np.ndarray] = []
        self.poses_new: List[np.ndarray] = []

    def add_points(self, points: np.ndarray, frame_id: int, pose: np.ndarray) -> None:
        """
        Add points in world frame. (Already transformed by pose when added.)
        frame_id: which pose was used to transform these points to world.
        pose: the pose at frame_id.
        """
        n = len(points)
        if points.shape[1] < 4:
            pts = np.column_stack([points[:, :3], np.zeros(n)])
        else:
            pts = points[:, :4]
        self.points = np.vstack([self.points, pts]) if self.points.size else pts
        self.frame_ids = (
            np.concatenate([self.frame_ids, np.full(n, frame_id)]) if self.frame_ids.size else np.full(n, frame_id)
        )
        while len(self.poses_old) <= frame_id:
            self.poses_old.append(np.eye(4))
            self.poses_new.append(np.eye(4))
        self.poses_old[frame_id] = pose.copy()
        self.poses_new[frame_id] = pose.copy()

    def set_poses(self, poses: List[np.ndarray]) -> None:
        """Set current poses (e.g. from odometry)."""
        self.poses_old = [p.copy() for p in poses]
        self.poses_new = [p.copy() for p in poses]

    def update_from_optimized_poses(
        self,
        optimized_poses: List[np.ndarray],
        affected_frame_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Apply pose correction: for each point in frame i,
        p_new = T_new[i] @ T_old[i]^{-1} @ p_old

        This is the FORM-style point-wise update.
        """
        if affected_frame_ids is None:
            affected_frame_ids = list(range(len(optimized_poses)))
        for fid in affected_frame_ids:
            if fid >= len(self.poses_old) or fid >= len(optimized_poses):
                continue
            T_old = self.poses_old[fid]
            T_new = optimized_poses[fid]
            mask = self.frame_ids == fid
            if not np.any(mask):
                continue
            # ΔT = T_new @ T_old^{-1}: transforms points from old world to new world
            delta_T = se3_compose(T_new, se3_inverse(T_old))
            pts = self.points[mask, :3]
            self.points[mask, :3] = (delta_T[:3, :3] @ pts.T).T + delta_T[:3, 3]
            self.poses_new[fid] = T_new.copy()
        self.poses_old = [p.copy() for p in self.poses_new]

    def get_points(self) -> np.ndarray:
        return self.points.copy()

    def get_point_count(self) -> int:
        return len(self.points)
