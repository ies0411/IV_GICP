#!/usr/bin/env python3
"""Pinpoint which exact frame seq02 IV-GICP diverges on."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_kitti_benchmark import load_kitti_sequence, kitti_gt_to_lidar_frame


def main():
    print("[seq02] loading full sequence...")
    frames_all, poses_gt = load_kitti_sequence("02", max_frames=2100)
    gt = kitti_gt_to_lidar_frame(poses_gt, "02")[:len(frames_all)]
    print(f"  {len(frames_all)} frames")

    from iv_gicp.pipeline import IVGICPPipeline
    p = IVGICPPipeline(
        voxel_size=1.0, source_voxel_size=0.3, min_points_per_voxel=3,
        alpha=0.1, max_correspondence_distance=2.0, initial_threshold=2.0,
        min_motion_th=0.1, max_map_points=100_000, max_map_frames=500,
        max_iterations=12, use_fim_weight=False, fim_auto_gate=0.0,
        auto_alpha=False, auto_alpha_from_intensity=False,
        source_drop_small_voxels=False, source_max_output_features=0,
        source_min_feature_score=0.0, max_source_points=0,
        map_radius=None, device="cpu",
    )

    prev_iv = np.eye(4)
    prev_gt = gt[0]
    for i, f in enumerate(frames_all):
        r = p.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        iv_pose = r.pose

        # per-frame relative motion
        dT_iv = np.linalg.inv(prev_iv) @ iv_pose
        dT_gt = np.linalg.inv(prev_gt) @ gt[i]

        # translation and rotation delta
        t_iv = np.linalg.norm(dT_iv[:3, 3])
        t_gt = np.linalg.norm(dT_gt[:3, 3])
        # rotation (angle in deg from rotation matrix)
        def rot_angle(R):
            c = (np.trace(R[:3, :3]) - 1) / 2.0
            c = max(-1.0, min(1.0, c))
            return np.degrees(np.arccos(c))
        r_iv = rot_angle(dT_iv)
        r_gt = rot_angle(dT_gt)

        # absolute position error
        pos_err = np.linalg.norm(iv_pose[:3, 3] - gt[i][:3, 3])

        if i >= 1700 and i <= 2050:
            print(f"  f={i:4d}  dT_iv={t_iv:6.3f}m dT_gt={t_gt:6.3f}m  "
                  f"dR_iv={r_iv:5.2f}° dR_gt={r_gt:5.2f}°  "
                  f"|err|={pos_err:7.3f}m  "
                  f"npts={len(f)}  pts={f[:, :3].min(axis=0).round(1)}→{f[:, :3].max(axis=0).round(1)}")
        elif i % 200 == 0:
            print(f"  f={i:4d}  |err|={pos_err:7.3f}m", end="\r", flush=True)

        prev_iv = iv_pose
        prev_gt = gt[i]


if __name__ == "__main__":
    main()
