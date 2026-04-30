#!/usr/bin/env python3
"""seq02 divergence localization: run IV-GICP at increasing horizons."""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_kitti_benchmark import load_kitti_sequence, kitti_gt_to_lidar_frame, ate_rmse, kitti_t_err


def run_iv(frames, mf, mr, mc, itr):
    from iv_gicp.pipeline import IVGICPPipeline
    p = IVGICPPipeline(
        voxel_size=1.0, source_voxel_size=0.3, min_points_per_voxel=3,
        alpha=0.1, max_correspondence_distance=mc, initial_threshold=2.0,
        min_motion_th=0.1, max_map_points=100_000, max_map_frames=mf,
        max_iterations=itr, use_fim_weight=False, fim_auto_gate=0.0,
        auto_alpha=False, auto_alpha_from_intensity=False,
        source_drop_small_voxels=False, source_max_output_features=0,
        source_min_feature_score=0.0, max_source_points=0,
        map_radius=mr, device="cpu",
    )
    poses = []
    for i, f in enumerate(frames):
        r = p.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        poses.append(r.pose.copy())
        if i % 500 == 0:
            print(f"    {i}/{len(frames)}", end="\r", flush=True)
    return poses


def main():
    print("[seq02] loading full sequence...")
    frames_all, poses_gt = load_kitti_sequence("02", max_frames=None)
    gt = kitti_gt_to_lidar_frame(poses_gt, "02")[:len(frames_all)]
    print(f"  {len(frames_all)} frames, {len(gt)} gt")

    horizons = [500, 1000, 1500, 2000, 3000, len(frames_all)]
    configs = [
        ("default",   dict(mf=500,  mr=None, mc=2.0, itr=12)),
        ("mf_large",  dict(mf=2000, mr=None, mc=2.0, itr=12)),
        ("mr_150",    dict(mf=500,  mr=150.0, mc=2.0, itr=12)),
        ("mr_100_mf_large", dict(mf=2000, mr=100.0, mc=2.0, itr=12)),
    ]

    for name, cfg in configs:
        print(f"\n[{name}] {cfg}")
        # full-seq run, then evaluate at each horizon
        t0 = time.perf_counter()
        poses = run_iv(frames_all, **cfg)
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        for h in horizons:
            if h > len(poses):
                continue
            ate, _ = ate_rmse(poses[:h], gt[:h])
            kt = kitti_t_err(poses[:h], gt[:h]) if h >= 200 else float("nan")
            print(f"  h={h:5d}  ATE={ate:8.3f}m  drift={kt:6.3f}%")


if __name__ == "__main__":
    main()
