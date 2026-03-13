#!/usr/bin/env python3
"""
Internal helper: run a single method and print JSON result.
Called by run_full_eval.py via subprocess for process isolation.

Usage:
  python examples/_run_single.py --method gicp_baseline --frames 15 --device cuda
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from iv_gicp.kitti_loader import load_kitti_sequence
from iv_gicp.metrics import compute_ate, compute_rpe, compute_rpe_kitti

def _eval(poses, poses_gt, frame_times, label, extra=None):
    result = {"label": label, "n_frames": len(poses)}
    if poses_gt and len(poses_gt) == len(poses):
        ate_rmse, ate_mean, _ = compute_ate(poses, poses_gt, align=True)
        rpe_rmse, rpe_mean    = compute_rpe(poses, poses_gt)
        kitti_m               = compute_rpe_kitti(poses, poses_gt)
        result.update({
            "ate_rmse": round(ate_rmse, 4), "ate_mean": round(ate_mean, 4),
            "rpe_rmse": round(rpe_rmse, 4), "rpe_mean": round(rpe_mean, 4),
            "kitti_t_err_pct":   round(kitti_m.get("t_err_pct", float("nan")), 4),
            "kitti_r_err_deg_m": round(kitti_m.get("r_err_deg_m", float("nan")), 6),
        })
    result["avg_frame_ms"]  = round(float(np.mean(frame_times) * 1000), 1)
    result["total_time_s"]  = round(float(sum(frame_times)), 2)
    if extra:
        result.update(extra)
    return result


def run_iv_gicp(frames, gt, label, device, **kwargs):
    from iv_gicp import IVGICPPipeline
    p = IVGICPPipeline(device=device, **kwargs)
    poses, ts = [], []
    for f in frames:
        t0 = time.perf_counter()
        p.process_frame(f)
        ts.append(time.perf_counter() - t0)
        poses.append(p.get_trajectory().poses[-1].copy())
    return _eval(poses, gt, ts, label)


def run_kiss(frames_xyz, gt):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.max_range = 80.0
    cfg.data.min_range = 2.0
    cfg.mapping.voxel_size = 1.0
    kiss = KissICP(config=cfg)
    poses, ts = [], []
    for pts in frames_xyz:
        t0 = time.perf_counter()
        kiss.register_frame(pts, np.zeros(len(pts)))
        ts.append(time.perf_counter() - t0)
        poses.append(kiss.last_pose.copy())
    return _eval(poses, gt, ts, "KISS-ICP")


def run_genz_proxy(frames, gt, device):
    """GenZ-ICP proxy: geometry-only + adaptive voxels + κ tracking."""
    from iv_gicp import IVGICPPipeline
    from iv_gicp.degeneracy_analysis import genz_icp_condition_number

    p = IVGICPPipeline(
        device=device, alpha=0.0, entropy_threshold=0.5,
        intensity_var_threshold=1e10, use_distribution_propagation=False,
        voxel_size=1.0, max_correspondence_distance=5.0,
        max_range=80.0, min_range=2.0, source_voxel_size=0.3, max_map_points=200_000,
    )
    poses, ts, kappas, plan_ratios = [], [], [], []
    for f in frames:
        t0 = time.perf_counter()
        p.process_frame(f)
        ts.append(time.perf_counter() - t0)
        poses.append(p.get_trajectory().poses[-1].copy())
        try:
            leaves = (p.flat_map.leaves if p.flat_map is not None else [])
            if leaves:
                covs = np.array([
                    getattr(v.stats, "cov", None) or getattr(v, "covariance", None)
                    for v in leaves
                    if (getattr(v, "stats", None) and getattr(v.stats, "cov", None) is not None)
                    or getattr(v, "covariance", None) is not None
                ])
                if len(covs) > 0:
                    H_t = np.sum(np.linalg.pinv(covs[:, :3, :3] + np.eye(3)*1e-6), axis=0)
                    H6  = np.block([[np.eye(3)*1e-3, np.zeros((3,3))],
                                    [np.zeros((3,3)), H_t]])
                    kappas.append(genz_icp_condition_number(H6))
                    eigs = np.linalg.eigvalsh(covs[:, :3, :3])
                    plan_ratios.append(float(np.mean(eigs[:, 0] / (eigs[:, -1] + 1e-10) < 0.1)))
        except Exception:
            pass

    extra = {}
    if kappas:
        extra["genz_kappa_mean"]    = round(float(np.mean(kappas)), 2)
        extra["genz_kappa_max"]     = round(float(np.max(kappas)), 2)
        extra["planarity_pct_mean"] = round(float(np.mean(plan_ratios)) * 100, 1)
        extra["genz_deg_frames"]    = int(np.sum(np.array(kappas) > 100))
    return _eval(poses, gt, ts, "GenZ-ICP (proxy)", extra)


# ─── Configs ─────────────────────────────────────────────────────────────────
COMMON = dict(
    voxel_size=1.0, max_correspondence_distance=5.0,
    max_range=80.0, min_range=2.0, source_voxel_size=0.3, max_map_points=200_000,
)
METHODS = {
    "gicp_baseline":  lambda f, g, d: run_iv_gicp(f, g, "GICP Baseline", d,
        alpha=0.0, entropy_threshold=1e10, intensity_var_threshold=1e10,
        use_distribution_propagation=False, **COMMON),
    "adaptive_only":  lambda f, g, d: run_iv_gicp(f, g, "+ Adaptive only", d,
        alpha=0.0, entropy_threshold=0.5, intensity_var_threshold=100.,
        use_distribution_propagation=False, **COMMON),
    "intensity_only": lambda f, g, d: run_iv_gicp(f, g, "+ Intensity only", d,
        alpha=0.1, entropy_threshold=1e10, intensity_var_threshold=1e10,
        use_distribution_propagation=False, **COMMON),
    "iv_gicp_no_dp":  lambda f, g, d: run_iv_gicp(f, g, "IV-GICP (no DP)", d,
        alpha=0.1, entropy_threshold=0.5, intensity_var_threshold=0.01,
        use_distribution_propagation=False, **COMMON),
    "iv_gicp_full":   lambda f, g, d: run_iv_gicp(f, g, "IV-GICP (Full)", d,
        alpha=0.1, entropy_threshold=0.5, intensity_var_threshold=0.01,
        use_distribution_propagation=True, **COMMON),
    "kiss_icp":       lambda f, g, d: run_kiss([x[:, :3] for x in f], g),
    "genz_proxy":     lambda f, g, d: run_genz_proxy(f, g, d),
}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--method",     required=True, choices=list(METHODS))
    pa.add_argument("--data",       default="data/kitti/sample")
    pa.add_argument("--frames",     type=int, default=15)
    pa.add_argument("--downsample", type=int, default=5)
    pa.add_argument("--device",     default="auto")
    args = pa.parse_args()

    frames, gt = load_kitti_sequence(args.data, args.frames, downsample=args.downsample)
    result = METHODS[args.method](frames, gt, args.device)
    # Serialize numpy types
    def _serial(o):
        if hasattr(o, "item"):
            return o.item()
        return o
    print(json.dumps({k: _serial(v) for k, v in result.items()}))


if __name__ == "__main__":
    main()
