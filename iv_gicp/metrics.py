"""
Odometry 평가 지표: ATE, RPE (논문용)

ATE: Absolute Trajectory Error (Umeyama SE(3) alignment 후 계산)
RPE: Relative Pose Error (consecutive frames + KITTI segment-based)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ─── Trajectory Alignment ─────────────────────────────────────────────────────

def align_trajectories_umeyama(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Align estimated trajectory to ground truth via Umeyama SE(3) alignment.
    Finds T_align = argmin Σ ||T_align · t_est_i - t_gt_i||²

    Returns:
        poses_aligned: estimated poses transformed to GT reference frame
        T_align: (4,4) alignment transform
    """
    n = min(len(poses_est), len(poses_gt))
    t_est = np.array([p[:3, 3] for p in poses_est[:n]])  # (n, 3)
    t_gt  = np.array([p[:3, 3] for p in poses_gt[:n]])   # (n, 3)

    mu_e = t_est.mean(axis=0)
    mu_g = t_gt.mean(axis=0)

    # Compute cross-covariance
    H = (t_est - mu_e).T @ (t_gt - mu_g)   # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection (ensure proper rotation det = +1)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1

    R_align = Vt.T @ D @ U.T
    t_align = mu_g - R_align @ mu_e

    T_align = np.eye(4)
    T_align[:3, :3] = R_align
    T_align[:3, 3] = t_align

    poses_aligned = [T_align @ p for p in poses_est[:n]]
    return poses_aligned, T_align


# ─── ATE ─────────────────────────────────────────────────────────────────────

def compute_ate(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
    align: bool = True,
) -> Tuple[float, float, np.ndarray]:
    """
    Absolute Trajectory Error (ATE) with optional SE(3) alignment.

    Args:
        poses_est: estimated poses (list of 4×4 matrices)
        poses_gt:  ground-truth poses (list of 4×4 matrices)
        align:     if True, perform Umeyama alignment before computing error
                   (required for odometry methods that have no global reference)

    Returns:
        (ATE_rmse, ATE_mean, per_frame_errors)
    """
    n = min(len(poses_est), len(poses_gt))
    if n < 2:
        return 0.0, 0.0, np.array([])

    if align:
        poses_aligned, _ = align_trajectories_umeyama(poses_est[:n], poses_gt[:n])
    else:
        poses_aligned = poses_est[:n]

    errs = np.array([
        np.linalg.norm(p[:3, 3] - gt[:3, 3])
        for p, gt in zip(poses_aligned, poses_gt[:n])
    ])
    return float(np.sqrt(np.mean(errs ** 2))), float(np.mean(errs)), errs


# ─── RPE (consecutive frames) ─────────────────────────────────────────────────

def compute_rpe(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
    delta: int = 1,
) -> Tuple[float, float]:
    """
    Relative Pose Error (RPE) for translation over frame stride δ.

    Error = ||ΔT_gt^{-1} · ΔT_est||_t  (translation part of relative transform error)

    Returns: (RPE_rmse, RPE_mean) in meters
    """
    n = min(len(poses_est), len(poses_gt))
    if n < 1 + delta:
        return 0.0, 0.0

    errs = []
    for i in range(n - delta):
        T_est_rel = np.linalg.inv(poses_est[i]) @ poses_est[i + delta]
        T_gt_rel  = np.linalg.inv(poses_gt[i])  @ poses_gt[i + delta]
        T_err = np.linalg.inv(T_gt_rel) @ T_est_rel
        errs.append(np.linalg.norm(T_err[:3, 3]))

    errs = np.array(errs)
    return float(np.sqrt(np.mean(errs ** 2))), float(np.mean(errs))


# ─── KITTI Odometry Standard RPE ─────────────────────────────────────────────

def compute_rpe_kitti(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
    step_size: int = 10,
    lengths: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    KITTI odometry benchmark evaluation: translation (%) and rotation (deg/m) errors
    over fixed distance segments (100m, 200m, ..., 800m).

    This matches the official KITTI devkit definition used for leaderboard comparison.

    Args:
        poses_est:  estimated trajectory (list of 4×4 matrices)
        poses_gt:   ground-truth trajectory (list of 4×4 matrices)
        step_size:  sample every step_size-th frame as segment start
        lengths:    segment lengths in meters to evaluate over

    Returns:
        dict with keys 't_err_pct' (mean translation error %) and
        'r_err_deg_m' (mean rotation error deg/m)
    """
    if lengths is None:
        lengths = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

    n = min(len(poses_est), len(poses_gt))
    if n < 2:
        return {"t_err_pct": 0.0, "r_err_deg_m": 0.0}

    # Precompute cumulative path length along GT trajectory
    cum_dist = np.zeros(n)
    for i in range(1, n):
        cum_dist[i] = cum_dist[i - 1] + np.linalg.norm(
            poses_gt[i][:3, 3] - poses_gt[i - 1][:3, 3]
        )

    t_errs: List[float] = []
    r_errs: List[float] = []

    for i in range(0, n, step_size):
        for length in lengths:
            target_dist = cum_dist[i] + length

            # Find the first frame j such that cumulative distance from i >= length
            j = np.searchsorted(cum_dist, target_dist)
            if j >= n:
                continue  # segment extends beyond trajectory

            actual_length = cum_dist[j] - cum_dist[i]
            if actual_length < 1.0:  # degenerate segment
                continue

            # Relative transforms over segment [i, j]
            T_est_seg = np.linalg.inv(poses_est[i]) @ poses_est[j]
            T_gt_seg  = np.linalg.inv(poses_gt[i])  @ poses_gt[j]
            T_err = np.linalg.inv(T_gt_seg) @ T_est_seg

            # Translation error (%)
            t_err_m = np.linalg.norm(T_err[:3, 3])
            t_errs.append(t_err_m / actual_length * 100.0)

            # Rotation error (deg/m)
            trace_val = np.clip((np.trace(T_err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            angle_rad = np.arccos(trace_val)
            r_errs.append(np.degrees(angle_rad) / actual_length)

    if not t_errs:
        return {"t_err_pct": 0.0, "r_err_deg_m": 0.0}

    return {
        "t_err_pct": float(np.mean(t_errs)),
        "r_err_deg_m": float(np.mean(r_errs)),
    }
