#!/usr/bin/env python3
"""
GenZ-ICP benchmark on GEODE Urban Tunnel sequences.

Extracts VLP-16 frames from ROS bags → temp .bin files → runs genz_bin_runner.
Outputs KITTI-format poses, computes ATE vs GEODE GT.

Usage:
    uv run python examples/bench_genz_geode.py
    uv run python examples/bench_genz_geode.py --seq 01 --max-frames 300
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

GEODE_ROOT   = Path("/home/km/data/GEODE")
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "geode"
RUNNER = Path(__file__).parent.parent / "thirdparty/genz-icp/bin_runner/build/genz_bin_runner"
LIDAR_TOPIC  = "/velodyne_points"

# Best params for GEODE Urban Tunnel: same as IV-GICP (0.5m voxel, 25m range)
GENZ_PARAMS = dict(
    voxel_size=0.5,
    max_range=25.0,
    min_range=0.5,
    min_motion_th=0.5,
    initial_threshold=1.5,
    planarity_th=0.1,
)


def extract_bag_to_bins(bag_path: Path, out_dir: Path, max_frames: int = None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps = []
    print(f"  Extracting: {bag_path.name}")
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = conns[0].msgcount if conns else 0
        count = 0
        for conn, ts_ns, rawdata in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            pts = _parse_vlp16(msg)
            if pts is None or len(pts) < 100:
                continue
            pts.astype(np.float32).tofile(out_dir / f"{count:06d}.bin")
            timestamps.append(ts_ns / 1e9)
            count += 1
            if count % 200 == 0:
                print(f"  {count}/{total} frames...", end="\r")
            if max_frames and count >= max_frames:
                break
    print(f"  {count} frames extracted    ")
    return timestamps


def _parse_vlp16(msg, max_range=25.0):
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 12:16].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    return pts[valid]


def load_geode_gt(gt_path: Path):
    from scipy.spatial.transform import Rotation
    data = np.loadtxt(gt_path)
    timestamps = data[:, 0]
    poses = []
    for row in data:
        R = Rotation.from_quat(row[4:8]).as_matrix()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = row[1:4]
        poses.append(T)
    T0_inv = np.linalg.inv(poses[0])
    return timestamps, [T0_inv @ p for p in poses]


def compute_ate(est_poses, est_timestamps, gt_timestamps, gt_poses):
    matched_est, matched_gt = [], []
    for i, ts in enumerate(est_timestamps):
        idx = np.argmin(np.abs(gt_timestamps - ts))
        if abs(gt_timestamps[idx] - ts) < 0.2:
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
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    return float(np.sqrt(np.mean(np.linalg.norm(R @ P + t - Q, axis=0)**2)))


def load_kitti_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.eye(4); T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def find_geode_paths(seq):
    """Try multiple possible GEODE data layouts."""
    seq_padded = seq.zfill(2)
    candidates = [
        (GEODE_ROOT / f"Urban_Tunnel{seq_padded}" / f"Urban_Tunnel{seq_padded}.bag",
         GEODE_ROOT / f"Urban_Tunnel{seq_padded}" / "gt.txt"),
        (GEODE_ROOT / "sensor_data" / "Urban_tunnel" / f"Urban_Tunnel{seq_padded}" / f"Urban_Tunnel{seq_padded}.bag",
         GEODE_ROOT / "groundtruth" / "Urban_tunnel" / f"Urban_Tunnel{seq_padded}.txt"),
    ]
    for bag, gt in candidates:
        if bag.exists():
            return bag, gt
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="01/02/03 or all")
    ap.add_argument("--max-frames", type=int, default=500)
    args = ap.parse_args()

    seqs = ["01", "02", "03"] if args.seq is None else [args.seq]

    # IV-GICP and KISS-ICP reference (CLAUDE.md 2026-03-22, 500fr)
    iv_ref   = {"Urban_Tunnel01": 2.706, "Urban_Tunnel02": 4.152, "Urban_Tunnel03": 12.528}
    kiss_ref = {"Urban_Tunnel01": 4.396, "Urban_Tunnel02": 8.085, "Urban_Tunnel03": 13.808}

    all_results = {}
    for seq in seqs:
        seq_name = f"Urban_Tunnel{seq.zfill(2)}"
        bag_path, gt_path = find_geode_paths(seq)
        if bag_path is None:
            print(f"[SKIP] {seq_name}: bag not found")
            continue

        out_dir = RESULTS_ROOT / seq_name
        out_dir.mkdir(parents=True, exist_ok=True)
        poses_out = out_dir / "genz_icp.txt"

        print(f"\n[{seq_name}]  bag={bag_path}")

        with tempfile.TemporaryDirectory(prefix="genz_geode_") as tmp:
            tmp_path = Path(tmp)
            timestamps = extract_bag_to_bins(bag_path, tmp_path, args.max_frames)

            cmd = [
                str(RUNNER), str(tmp_path), str(poses_out),
                "--max-frames", str(args.max_frames),
                "--voxel-size",   str(GENZ_PARAMS["voxel_size"]),
                "--max-range",    str(GENZ_PARAMS["max_range"]),
                "--min-range",    str(GENZ_PARAMS["min_range"]),
                "--min-motion-th",str(GENZ_PARAMS["min_motion_th"]),
                "--initial-threshold", str(GENZ_PARAMS["initial_threshold"]),
                "--planarity-th", str(GENZ_PARAMS["planarity_th"]),
            ]
            print(f"  {RUNNER.name} params: voxel={GENZ_PARAMS['voxel_size']}"
                  f"  max_range={GENZ_PARAMS['max_range']}"
                  f"  min_th={GENZ_PARAMS['min_motion_th']}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout[-600:] if res.stdout else "")
            if res.returncode != 0:
                print(f"  ERROR: {res.stderr[-300:]}")
                continue

        est_poses = load_kitti_poses(poses_out)
        if not gt_path.exists():
            print(f"  [WARN] GT not found: {gt_path}")
            continue
        gt_ts, gt_poses = load_geode_gt(gt_path)
        ate = compute_ate(est_poses, timestamps[:len(est_poses)], gt_ts, gt_poses)
        iv  = iv_ref.get(seq_name, float('nan'))
        ki  = kiss_ref.get(seq_name, float('nan'))
        print(f"  GenZ ATE={ate:.3f}m  IV-GICP={iv:.3f}m  KISS={ki:.3f}m  ({len(est_poses)} fr)")
        all_results[seq_name] = {"genz_icp": ate, "iv_gicp": iv, "kiss_icp": ki}

    # Summary table
    print(f"\n{'='*60}")
    print(f"GEODE Urban Tunnel  {args.max_frames}fr   GenZ-ICP vs IV-GICP vs KISS-ICP")
    print(f"{'Seq':<20} {'IV-GICP':>9} {'KISS-ICP':>10} {'GenZ-ICP':>10}")
    print("-" * 55)
    for seq_name, r in all_results.items():
        print(f"{seq_name:<20} {r['iv_gicp']:>9.3f}  {r['kiss_icp']:>10.3f}  {r['genz_icp']:>10.3f}")

    out_json = RESULTS_ROOT / "genz_results.json"
    with open(out_json, "w") as f:
        json.dump({"params": GENZ_PARAMS, "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_json}")


if __name__ == "__main__":
    main()
