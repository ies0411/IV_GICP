#!/usr/bin/env python3
"""Diagnose SubT Final_UGV1 full-seq drift growth.

Runs IV-GICP on SubT Final_UGV1 and saves poses every frame. Computes
Umeyama-aligned ATE at checkpoints to see whether drift grows steadily
(slow leak → map-management issue) or spikes at a specific frame
(single failure mode → min_motion_th / adaptive threshold issue).

Usage:
  python examples/diag_subt_fullseq_drift.py --min-th 0.5
  python examples/diag_subt_fullseq_drift.py --min-th 1.0
"""
import argparse
import sys
import time
import os
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_subt_eval import (
    DATASETS, BASE_DIR,
    load_frames_from_zipped_bags,
    load_gt_csv,
    compute_ate,
)
from iv_gicp.pipeline import IVGICPPipeline


CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000, 99999]


def umeyama_ate(est_T, gt_t):
    """Umeyama-aligned ATE given paired translations."""
    P = np.array([T[:3, 3] for T in est_T]).T  # 3xN
    Q = np.array(gt_t).T
    if P.shape[1] < 5:
        return float("nan")
    mu_p = P.mean(axis=1, keepdims=True)
    mu_q = Q.mean(axis=1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    errs = np.linalg.norm(R @ P + t - Q, axis=0)
    return float(np.sqrt(np.mean(errs ** 2)))


def match_to_gt(est_poses, est_ts, gt_ts, gt_poses, win_ms=500):
    matched_est, matched_gt = [], []
    for i, ts in enumerate(est_ts):
        idx = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(int(gt_ts[idx]) - int(ts)) / 1e6 < win_ms:
            matched_est.append(est_poses[i])
            matched_gt.append(gt_poses[idx][:3, 3])
    return matched_est, matched_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Final_UGV1")
    ap.add_argument("--min-th", type=float, default=0.5)
    ap.add_argument("--max-map-frames", type=int, default=200)
    ap.add_argument("--map-radius", type=float, default=200.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    info = DATASETS[args.dataset]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])
    gt_zip = os.path.join(BASE_DIR, info["gt"])

    print(f"[diag] dataset={args.dataset}  min_th={args.min_th}  "
          f"mf={args.max_map_frames}  mr={args.map_radius}  α={args.alpha}")

    print("[diag] Loading GT...")
    gt_ts, gt_poses = load_gt_csv(gt_zip, info["gt_file"])
    print(f"  GT: {len(gt_poses)} poses")

    pipeline = IVGICPPipeline(
        voxel_size=0.5,
        source_voxel_size=0.3,
        alpha=args.alpha,
        max_correspondence_distance=2.0,
        initial_threshold=1.5,
        max_iterations=12,
        huber_delta=1.0,
        min_range=0.5,
        max_range=80.0,
        max_map_frames=args.max_map_frames,
        map_radius=args.map_radius,
        min_motion_th=args.min_th,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        device="cpu",
    )

    poses, ts_list = [], []
    checkpoints_done = {}

    t_start = time.perf_counter()
    for i, (ts_ns, cloud) in enumerate(load_frames_from_zipped_bags(rosbag_zip)):
        r = pipeline.process_frame(cloud[:, :3], cloud[:, 3], timestamp=float(i))
        poses.append(r.pose.copy())
        ts_list.append(ts_ns)

        if (i + 1) in CHECKPOINTS or (i + 1) % 1000 == 0:
            # Compute ATE up to current frame
            est = poses[: i + 1]
            ts_arr = np.array(ts_list[: i + 1])
            me, mg = match_to_gt(est, ts_arr, gt_ts, gt_poses)
            if len(me) >= 5:
                ate = umeyama_ate(me, mg)
            else:
                ate = float("nan")
            path_len = sum(
                np.linalg.norm(est[j + 1][:3, 3] - est[j][:3, 3])
                for j in range(len(est) - 1)
            )
            elapsed = time.perf_counter() - t_start
            print(
                f"  [ckpt] i={i+1:5d}  ATE={ate:7.3f}m  path={path_len:7.1f}m  "
                f"drift%={100.0*ate/path_len if path_len>0 else 0:5.3f}  ({elapsed:.0f}s)",
                flush=True,
            )
            checkpoints_done[i + 1] = {
                "ate": ate, "path": path_len,
                "drift_pct": 100.0 * ate / path_len if path_len > 0 else 0.0,
            }

    # Final report
    out_path = Path("results/fullseq_logs") / f"subt_{args.dataset}_drift_minth{args.min_th}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"min_th": args.min_th, "mf": args.max_map_frames,
         "mr": args.map_radius, "alpha": args.alpha,
         "n_frames": len(poses),
         "checkpoints": checkpoints_done},
        indent=2,
    ))
    print(f"\n[diag] Saved → {out_path}")


if __name__ == "__main__":
    main()
