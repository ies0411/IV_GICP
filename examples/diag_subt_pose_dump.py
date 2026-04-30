#!/usr/bin/env python3
"""Dump all IV-GICP per-frame poses on SubT Final_UGV1.

Output: results/fullseq_logs/subt_Final_UGV1_poses_minth{X}.npz
  - poses: (N, 4, 4)
  - timestamps_ns: (N,)
  - per_frame_reg_ms: (N,)

Use for post-hoc analysis of fr 12500-13000 divergence event.
"""
import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_subt_eval import DATASETS, BASE_DIR, load_frames_from_zipped_bags
from iv_gicp.pipeline import IVGICPPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Final_UGV1")
    ap.add_argument("--min-th", type=float, default=0.5)
    ap.add_argument("--out-dir", default="results/fullseq_logs")
    args = ap.parse_args()

    info = DATASETS[args.dataset]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])

    print(f"[pose-dump] {args.dataset}  min_th={args.min_th}")

    pipeline = IVGICPPipeline(
        voxel_size=0.5,
        source_voxel_size=0.3,
        alpha=0.1,
        max_correspondence_distance=2.0,
        initial_threshold=1.5,
        max_iterations=12,
        huber_delta=1.0,
        min_range=0.5,
        max_range=80.0,
        max_map_frames=200,
        map_radius=200.0,
        min_motion_th=args.min_th,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        device="cpu",
    )

    poses, ts_list, reg_ms = [], [], []
    t0 = time.perf_counter()

    for i, (ts_ns, cloud) in enumerate(load_frames_from_zipped_bags(rosbag_zip)):
        r = pipeline.process_frame(cloud[:, :3], cloud[:, 3], timestamp=float(i))
        poses.append(r.pose.copy())
        ts_list.append(ts_ns)
        reg_ms.append(getattr(r, "reg_ms", 0.0) or 0.0)

        if (i + 1) % 500 == 0:
            print(f"  frame {i+1}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    out_path = Path(args.out_dir) / f"subt_{args.dataset}_poses_minth{args.min_th}.npz"
    np.savez(
        out_path,
        poses=np.array(poses),
        timestamps_ns=np.array(ts_list, dtype=np.int64),
        reg_ms=np.array(reg_ms),
    )
    print(f"\n[pose-dump] Saved {len(poses)} poses → {out_path}")


if __name__ == "__main__":
    main()
