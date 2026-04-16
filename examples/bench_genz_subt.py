#!/usr/bin/env python3
"""
GenZ-ICP benchmark on SubT-MRS sequences.

Reuses SubT frame loader from run_subt_eval.py, saves as .bin, runs genz_bin_runner.

Usage:
    uv run python examples/bench_genz_subt.py
    uv run python examples/bench_genz_subt.py --dataset Final_UGV1 --max-frames 300
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = "/home/km/deepai_dev_data/SubT-MRS"
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "subt"
RUNNER = Path(__file__).parent.parent / "thirdparty/genz-icp/bin_runner/build/genz_bin_runner"

DATASETS = {
    "Urban_UGV1":  {"rosbag": "rosbag/SubT_MRS_Urban_Challenge_UGV1.zip",
                    "gt": "LiDAR_Inertial_Track/SubT_MRS_Urban_Challenge_UGV1.zip",
                    "gt_file": "SubT_MRS_Urban_Challenge_UGV1/ground_truth_path.csv"},
    "Urban_UGV2":  {"rosbag": "rosbag/SubT_MRS_Urban_Challenge_UGV2.zip",
                    "gt": "LiDAR_Inertial_Track/SubT_MRS_Urban_Challenge_UGV2.zip",
                    "gt_file": "SubT_MRS_Urban_Challenge_UGV2/ground_truth_path.csv"},
    "Final_UGV1":  {"rosbag": "rosbag/SubT_MRS_Final_Challenge_UGV1.zip",
                    "gt": "LiDAR_Inertial_Track/SubT_MRS_Final_Challenge_UGV1.zip",
                    "gt_file": "SubT_MRS_Final_Challenge_UGV1/ground_truth_path.csv"},
    "Final_UGV2":  {"rosbag": "rosbag/SubT_MRS_Final_Challenge_UGV2.zip",
                    "gt": "LiDAR_Inertial_Track/SubT_MRS_Final_Challenge_UGV2.zip",
                    "gt_file": "SubT_MRS_Final_Challenge_UGV2/ground_truth_path.csv"},
    "Final_UGV3":  {"rosbag": "rosbag/SubT_MRS_Final_Challenge_UGV3.zip",
                    "gt": "LiDAR_Inertial_Track/SubT_MRS_Final_Challenge_UGV3.zip",
                    "gt_file": "SubT_MRS_Final_Challenge_UGV3/ground_truth_path.csv"},
}

# SubT tunnel params (same as IV-GICP best)
GENZ_PARAMS = dict(
    voxel_size=0.5,
    max_range=50.0,
    min_range=0.5,
    min_motion_th=0.5,
    initial_threshold=1.5,
    planarity_th=0.1,
)


def load_frames_as_bins(dataset: str, max_frames: int, out_dir: Path):
    """Load SubT frames using run_subt_eval loader, save as .bin."""
    from run_subt_eval import load_frames_from_zipped_bags
    import os
    info = DATASETS[dataset]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])
    if not Path(rosbag_zip).exists():
        return [], []

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps = []
    count = 0
    print(f"  Loading SubT frames from {Path(rosbag_zip).name}...")
    for ts_ns, pts in load_frames_from_zipped_bags(rosbag_zip, max_frames):
        if pts is None or len(pts) < 100:
            continue
        # pts: (N,4) float64 x,y,z,intensity — save as float32
        pts.astype(np.float32).tofile(out_dir / f"{count:06d}.bin")
        timestamps.append(ts_ns)
        count += 1
        if count % 100 == 0:
            print(f"  {count} frames loaded...", end="\r")
    print(f"\n  {count} frames total")
    return timestamps, info


def load_gt_csv(dataset: str):
    from run_subt_eval import load_gt_csv
    import os
    info = DATASETS[dataset]
    gt_zip = os.path.join(BASE_DIR, info["gt"])
    if not Path(gt_zip).exists():
        return None, None
    return load_gt_csv(gt_zip, info["gt_file"])


def compute_ate(est_poses, est_ts_ns, gt_ts_ns, gt_poses):
    est_ts = np.array(est_ts_ns, dtype=np.int64)
    matched_est, matched_gt = [], []
    for i, ts in enumerate(est_ts):
        idx = np.argmin(np.abs(gt_ts_ns - ts))
        dt_ms = abs(int(gt_ts_ns[idx]) - int(ts)) / 1e6
        if dt_ms < 200:
            matched_est.append(est_poses[i][:3, 3])
            matched_gt.append(gt_poses[idx][:3, 3])
    if len(matched_est) < 5:
        return float('nan')
    P = np.array(matched_est).T
    Q = np.array(matched_gt).T
    mu_p = P.mean(1, keepdims=True); mu_q = Q.mean(1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_mat = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R_mat @ mu_p
    return float(np.sqrt(np.mean(np.linalg.norm(R_mat @ P + t - Q, axis=0)**2)))


def load_kitti_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.eye(4); T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--max-frames", type=int, default=500)
    args = ap.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset is None else [args.dataset]

    # IV-GICP and KISS-ICP reference (CLAUDE.md 2026-03-22, 500fr)
    iv_ref   = {"Urban_UGV1": 0.276, "Urban_UGV2": 0.280, "Final_UGV1": 0.084,
                "Final_UGV2": 0.031, "Final_UGV3": 0.014}
    kiss_ref = {"Urban_UGV1": 0.285, "Urban_UGV2": 0.288, "Final_UGV1": 0.088,
                "Final_UGV2": 0.031, "Final_UGV3": 0.016}

    all_results = {}
    for ds in datasets:
        out_dir = RESULTS_ROOT / ds
        poses_out = out_dir / "genz_icp.txt"

        print(f"\n[{ds}]")
        with tempfile.TemporaryDirectory(prefix="genz_subt_") as tmp:
            tmp_path = Path(tmp)
            timestamps, info = load_frames_as_bins(ds, args.max_frames, tmp_path)
            if not timestamps:
                print(f"  [SKIP] no data")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(RUNNER), str(tmp_path), str(poses_out),
                "--max-frames",   str(args.max_frames),
                "--voxel-size",   str(GENZ_PARAMS["voxel_size"]),
                "--max-range",    str(GENZ_PARAMS["max_range"]),
                "--min-range",    str(GENZ_PARAMS["min_range"]),
                "--min-motion-th",str(GENZ_PARAMS["min_motion_th"]),
                "--initial-threshold", str(GENZ_PARAMS["initial_threshold"]),
                "--planarity-th", str(GENZ_PARAMS["planarity_th"]),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout[-400:] if res.stdout else "")
            if res.returncode != 0:
                print(f"  ERROR: {res.stderr[-200:]}")
                continue

        est_poses = load_kitti_poses(poses_out)
        gt_ts, gt_poses = load_gt_csv(ds)
        if gt_ts is None:
            print(f"  [WARN] GT not found")
            continue
        ate = compute_ate(est_poses, timestamps[:len(est_poses)], gt_ts, gt_poses)
        iv  = iv_ref.get(ds, float('nan'))
        ki  = kiss_ref.get(ds, float('nan'))
        print(f"  GenZ ATE={ate:.4f}m  IV-GICP={iv:.4f}m  KISS={ki:.4f}m")
        all_results[ds] = {"genz_icp": ate, "iv_gicp": iv, "kiss_icp": ki}

    print(f"\n{'='*60}")
    print(f"SubT-MRS {args.max_frames}fr   GenZ-ICP vs IV-GICP vs KISS-ICP")
    print(f"{'Dataset':<18} {'IV-GICP':>9} {'KISS-ICP':>10} {'GenZ-ICP':>10}")
    print("-" * 52)
    for ds, r in all_results.items():
        print(f"{ds:<18} {r['iv_gicp']:>9.4f}  {r['kiss_icp']:>10.4f}  {r['genz_icp']:>10.4f}")

    out_json = RESULTS_ROOT / "genz_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"params": GENZ_PARAMS, "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_json}")


if __name__ == "__main__":
    main()
