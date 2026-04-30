#!/usr/bin/env python3
"""Diagnose MulRan DCC01 full-seq divergence — param sweep.

min_th=0.5 alone failed to fix MulRan (39.19m → 38.77m = -1%).
Try other knobs: max_map_frames, max_correspondence, alpha.

Runs IV-GICP with each config and reports final + checkpoint ATE.
"""
import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_mulran_eval import load_sequence, compute_ate
from iv_gicp.pipeline import IVGICPPipeline


CHECKPOINTS = [500, 1000, 2000, 3000, 4000, 5000, 99999]


def umeyama_ate(est_T, gt_t):
    P = np.array([T[:3, 3] for T in est_T]).T
    Q = np.array(gt_t).T
    if P.shape[1] < 5:
        return float("nan")
    mu_p, mu_q = P.mean(1, keepdims=True), Q.mean(1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    errs = np.linalg.norm(R @ P + t - Q, axis=0)
    return float(np.sqrt(np.mean(errs ** 2)))


def match_to_gt(est_poses, est_ts_s, gt_ts_s, gt_poses, win_s=0.5):
    matched_est, matched_gt = [], []
    for i, ts in enumerate(est_ts_s):
        idx = int(np.argmin(np.abs(gt_ts_s - ts)))
        if abs(float(gt_ts_s[idx]) - float(ts)) < win_s:
            matched_est.append(est_poses[i])
            matched_gt.append(gt_poses[idx][:3, 3])
    return matched_est, matched_gt


def run_one(seq, min_th, mf, mc, alpha, label):
    print(f"\n=== {label}: min_th={min_th} mf={mf} mc={mc} α={alpha} ===")
    frames, gt_ts, gt_poses, scan_ts = load_sequence(seq)
    print(f"  {len(frames)} frames, {len(gt_poses)} GT poses")

    pipeline = IVGICPPipeline(
        voxel_size=1.0,
        source_voxel_size=0.3,
        alpha=alpha,
        max_correspondence_distance=mc,
        initial_threshold=2.0,
        min_motion_th=min_th,
        max_iterations=20,
        max_map_frames=mf,
        map_radius=None,
        max_source_points=0,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        device="cpu",
    )

    poses = []
    scan_ts_s = scan_ts / 1e9 if scan_ts.dtype == np.int64 else scan_ts
    gt_ts_s = gt_ts / 1e9 if gt_ts.dtype == np.int64 else gt_ts
    checkpoints_done = {}
    t0 = time.perf_counter()

    for i, pts in enumerate(frames):
        r = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=float(i))
        poses.append(r.pose.copy())

        if (i + 1) in CHECKPOINTS or (i + 1) % 1000 == 0:
            est_ts_cur = scan_ts_s[: i + 1]
            me, mg = match_to_gt(poses, est_ts_cur, gt_ts_s, gt_poses)
            ate = umeyama_ate(me, mg) if len(me) >= 5 else float("nan")
            path = sum(
                np.linalg.norm(poses[j + 1][:3, 3] - poses[j][:3, 3])
                for j in range(len(poses) - 1)
            )
            print(
                f"  [ckpt] i={i+1:4d}  ATE={ate:8.3f}m  path={path:7.1f}m  "
                f"drift%={100*ate/path if path>0 else 0:5.2f}  "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
            checkpoints_done[i + 1] = {
                "ate": ate, "path": path,
                "drift_pct": 100 * ate / path if path > 0 else 0,
            }

    return checkpoints_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="DCC01")
    ap.add_argument("--config", default="baseline",
                    choices=["baseline", "mf100", "mf50", "mc1.0",
                             "mc0.8", "minth1.0", "alpha0"])
    args = ap.parse_args()

    configs = {
        "baseline":  dict(min_th=0.5, mf=500, mc=2.0, alpha=0.1),  # current fix attempt
        "mf100":     dict(min_th=0.5, mf=100, mc=2.0, alpha=0.1),
        "mf50":      dict(min_th=0.5, mf=50,  mc=2.0, alpha=0.1),
        "mc1.0":     dict(min_th=0.5, mf=500, mc=1.0, alpha=0.1),
        "mc0.8":     dict(min_th=0.5, mf=500, mc=0.8, alpha=0.1),
        "minth1.0":  dict(min_th=1.0, mf=500, mc=2.0, alpha=0.1),
        "alpha0":    dict(min_th=0.5, mf=500, mc=2.0, alpha=0.0),
    }
    cfg = configs[args.config]

    result = run_one(args.seq, label=args.config, **cfg)

    out = Path("results/fullseq_logs") / f"mulran_{args.seq}_sweep_{args.config}.json"
    out.write_text(json.dumps({"config": args.config, **cfg,
                                "checkpoints": result}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
