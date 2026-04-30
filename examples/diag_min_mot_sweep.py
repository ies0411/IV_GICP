#!/usr/bin/env python3
"""Test min_motion_th sweep on seq01, seq02 full + GEODE Urban Tunnel 500fr non-regression."""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def kitti_load(seq, max_frames=None):
    from run_kitti_benchmark import load_kitti_sequence, kitti_gt_to_lidar_frame, ate_rmse, kitti_t_err
    frames, poses_gt = load_kitti_sequence(seq, max_frames=max_frames)
    gt = kitti_gt_to_lidar_frame(poses_gt, seq)[:len(frames)]
    return frames, gt, ate_rmse, kitti_t_err


def geode_load(seq, max_frames):
    from run_geode_eval import load_geode_sequence
    return load_geode_sequence(seq, max_frames=max_frames)


def run_iv(frames, min_mot, tunnel=False):
    from iv_gicp.pipeline import IVGICPPipeline
    if tunnel:
        p = IVGICPPipeline(
            voxel_size=0.5, source_voxel_size=0.25, min_points_per_voxel=3,
            alpha=0.0, max_correspondence_distance=2.0, initial_threshold=2.0,
            min_motion_th=min_mot, max_map_points=100_000, max_map_frames=500,
            max_iterations=12, use_fim_weight=True, fim_auto_gate=0.0,
            auto_alpha=False, auto_alpha_from_intensity=False,
            source_drop_small_voxels=False, source_max_output_features=0,
            source_min_feature_score=0.0, max_source_points=0,
            map_radius=80.0, device="cpu",
        )
    else:
        p = IVGICPPipeline(
            voxel_size=1.0, source_voxel_size=0.3, min_points_per_voxel=3,
            alpha=0.1, max_correspondence_distance=2.0, initial_threshold=2.0,
            min_motion_th=min_mot, max_map_points=100_000, max_map_frames=500,
            max_iterations=12, use_fim_weight=False, fim_auto_gate=0.0,
            auto_alpha=False, auto_alpha_from_intensity=False,
            source_drop_small_voxels=False, source_max_output_features=0,
            source_min_feature_score=0.0, max_source_points=0,
            map_radius=None, device="cpu",
        )
    poses = []
    for i, f in enumerate(frames):
        r = p.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        poses.append(r.pose.copy())
        if i % 500 == 0:
            print(f"    {i}/{len(frames)}", end="\r", flush=True)
    return poses


def ate_rmse_simple(poses, gt):
    # Umeyama-aligned ATE
    t_est = np.array([P[:3, 3] for P in poses])
    t_gt = np.array([G[:3, 3] for G in gt])
    mu_e, mu_g = t_est.mean(0), t_gt.mean(0)
    t_est_c = t_est - mu_e
    t_gt_c = t_gt - mu_g
    H = t_est_c.T @ t_gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t_aligned = t_est @ R.T + (mu_g - R @ mu_e)
    err = np.linalg.norm(t_aligned - t_gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    sweep = [0.1, 0.3, 0.5, 0.7, 1.0]
    results = {}

    # KITTI seq01 full
    print("\n=== KITTI seq01 full (1101 fr) ===")
    frames, gt, _, kitti_t_err = kitti_load("01")
    print(f"  {len(frames)} frames")
    for m in sweep:
        print(f"\n[min_mot={m}]")
        t0 = time.perf_counter()
        poses = run_iv(frames, min_mot=m, tunnel=False)
        dt = time.perf_counter() - t0
        ate = ate_rmse_simple(poses, gt)
        kt = kitti_t_err(poses, gt)
        print(f"  ATE={ate:8.3f}m  drift={kt:6.3f}%  ({dt:.0f}s)")
        results[f"seq01_{m}"] = {"ate": ate, "drift": kt, "time_s": dt}

    # KITTI seq02 full
    print("\n=== KITTI seq02 full (4661 fr) ===")
    frames, gt, _, kitti_t_err = kitti_load("02")
    print(f"  {len(frames)} frames")
    for m in sweep:
        print(f"\n[min_mot={m}]")
        t0 = time.perf_counter()
        poses = run_iv(frames, min_mot=m, tunnel=False)
        dt = time.perf_counter() - t0
        ate = ate_rmse_simple(poses, gt)
        kt = kitti_t_err(poses, gt)
        print(f"  ATE={ate:8.3f}m  drift={kt:6.3f}%  ({dt:.0f}s)")
        results[f"seq02_{m}"] = {"ate": ate, "drift": kt, "time_s": dt}

    # GEODE tunnel test skipped here; will run run_geode_eval.py directly after

    print("\n\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:15s}  ATE={v['ate']:8.3f}  drift={v.get('drift', 0):6.3f}%")


if __name__ == "__main__":
    main()
