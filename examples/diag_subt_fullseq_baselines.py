#!/usr/bin/env python3
"""SubT Final_UGV1 full-seq baseline comparison (KISS + GenZ).

Runs KISS-ICP and GenZ-ICP on the full 16747-frame sequence with
per-checkpoint ATE so we can compare to IV-GICP's drift profile
(subt_Final_UGV1_drift_minth0.5.json).

Usage:
    uv run python examples/diag_subt_fullseq_baselines.py --method kiss
    uv run python examples/diag_subt_fullseq_baselines.py --method genz
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_subt_eval import DATASETS, BASE_DIR, load_frames_from_zipped_bags, load_gt_csv

CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 7500, 10000,
               12000, 12500, 13000, 14000, 15000, 99999]


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
    return float(np.sqrt(np.mean(np.linalg.norm(R @ P + t - Q, axis=0) ** 2)))


def match_to_gt(est_poses, est_ts, gt_ts, gt_poses, win_ms=500):
    me, mg = [], []
    for i, ts in enumerate(est_ts):
        idx = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(int(gt_ts[idx]) - int(ts)) / 1e6 < win_ms:
            me.append(est_poses[i])
            mg.append(gt_poses[idx][:3, 3])
    return me, mg


def run_kiss(dataset, rosbag_zip):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.min_range = 0.5
    cfg.data.max_range = 50.0
    cfg.data.deskew = False
    cfg.mapping.voxel_size = 0.5
    od = KissICP(config=cfg)
    poses, ts_list = [], []

    t0 = time.perf_counter()
    ckpts = {}
    for i, (ts_ns, cloud) in enumerate(load_frames_from_zipped_bags(rosbag_zip)):
        od.register_frame(cloud[:, :3].astype(np.float64),
                          np.full(len(cloud), float(i)))
        poses.append(od.last_pose.copy())
        ts_list.append(ts_ns)
        if (i + 1) in CHECKPOINTS or (i + 1) % 1000 == 0:
            yield i + 1, poses, ts_list, time.perf_counter() - t0


def run_genz(dataset, rosbag_zip, max_frames=99999):
    """Run GenZ via bin_runner subprocess. Yields once at end with full results."""
    RUNNER = Path(__file__).parent.parent / "thirdparty/genz-icp/bin_runner/build/genz_bin_runner"
    with tempfile.TemporaryDirectory(prefix="genz_subt_fs_") as tmp:
        tmp_path = Path(tmp)
        ts_list = []
        count = 0
        print(f"  loading frames → bin...")
        for ts_ns, pts in load_frames_from_zipped_bags(rosbag_zip, max_frames):
            pts.astype(np.float32).tofile(tmp_path / f"{count:06d}.bin")
            ts_list.append(ts_ns)
            count += 1
            if count % 1000 == 0:
                print(f"    {count} frames loaded...")
        print(f"  {count} total frames, running GenZ...")

        poses_out = tmp_path / "genz_poses.txt"
        cmd = [str(RUNNER), str(tmp_path), str(poses_out),
               "--max-frames", str(count),
               "--voxel-size", "0.5", "--max-range", "50.0", "--min-range", "0.5",
               "--min-motion-th", "0.5", "--initial-threshold", "1.5",
               "--planarity-th", "0.1"]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if res.returncode != 0:
            print(f"  ERROR: {res.stderr[-500:]}")
            return

        # Load poses (KITTI 3x4 format)
        poses = []
        with open(poses_out) as f:
            for line in f:
                vals = list(map(float, line.split()))
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
        yield len(poses), poses, ts_list[:len(poses)], elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["kiss", "genz"])
    ap.add_argument("--dataset", default="Final_UGV1")
    args = ap.parse_args()

    info = DATASETS[args.dataset]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])
    gt_zip = os.path.join(BASE_DIR, info["gt"])
    gt_ts, gt_poses = load_gt_csv(gt_zip, info["gt_file"])
    print(f"[baseline] {args.method}  {args.dataset}  GT poses={len(gt_poses)}")

    ckpt_data = {}
    runner = run_kiss if args.method == "kiss" else run_genz
    for n, poses, ts_list, elapsed in runner(args.dataset, rosbag_zip):
        ts_arr = np.array(ts_list)
        me, mg = match_to_gt(poses, ts_arr, gt_ts, gt_poses)
        ate = umeyama_ate(me, mg) if len(me) >= 5 else float("nan")
        path = sum(np.linalg.norm(poses[j+1][:3,3]-poses[j][:3,3])
                   for j in range(len(poses)-1))
        drift = 100.0 * ate / path if path > 0 else 0
        print(f"  [ckpt] i={n:5d}  ATE={ate:7.3f}m  path={path:7.1f}m  "
              f"drift%={drift:5.3f}  ({elapsed:.0f}s)", flush=True)
        ckpt_data[n] = {"ate": ate, "path": path, "drift_pct": drift}

    out = Path("results/fullseq_logs") / f"subt_{args.dataset}_{args.method}_drift.json"
    out.write_text(json.dumps({"method": args.method, "checkpoints": ckpt_data}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
