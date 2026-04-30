#!/usr/bin/env python3
"""seq02 v2: test alpha, correspondence distance, min_motion, fim_auto_gate."""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_kitti_benchmark import load_kitti_sequence, kitti_gt_to_lidar_frame, ate_rmse, kitti_t_err


def run_iv(frames, **overrides):
    from iv_gicp.pipeline import IVGICPPipeline
    params = dict(
        voxel_size=1.0, source_voxel_size=0.3, min_points_per_voxel=3,
        alpha=0.1, max_correspondence_distance=2.0, initial_threshold=2.0,
        min_motion_th=0.1, max_map_points=100_000, max_map_frames=500,
        max_iterations=12, use_fim_weight=False, fim_auto_gate=0.0,
        auto_alpha=False, auto_alpha_from_intensity=False,
        source_drop_small_voxels=False, source_max_output_features=0,
        source_min_feature_score=0.0, max_source_points=0,
        map_radius=None, device="cpu",
    )
    params.update(overrides)
    p = IVGICPPipeline(**params)
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
    print(f"  {len(frames_all)} frames")

    horizons = [500, 1000, 1500, 1800, 2000, 2500, 3000, len(frames_all)]
    configs = [
        ("alpha_0",      dict(alpha=0.0)),
        ("mc_1p5",       dict(max_correspondence_distance=1.5)),
        ("mc_3p0",       dict(max_correspondence_distance=3.0)),
        ("min_mot_0p5",  dict(min_motion_th=0.5)),
        ("itr_20",       dict(max_iterations=20)),
        ("itr_8",        dict(max_iterations=8)),
        ("init_th_1",    dict(initial_threshold=1.0)),
    ]

    for name, cfg in configs:
        print(f"\n[{name}] {cfg}")
        t0 = time.perf_counter()
        poses = run_iv(frames_all, **cfg)
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        for h in horizons:
            if h > len(poses):
                continue
            ate, _ = ate_rmse(poses[:h], gt[:h])
            kt = kitti_t_err(poses[:h], gt[:h]) if h >= 800 else float("nan")
            print(f"  h={h:5d}  ATE={ate:8.3f}m  drift={kt:6.3f}%")


if __name__ == "__main__":
    main()
