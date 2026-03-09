#!/usr/bin/env python3
"""
IV-GICP Ablation Study: C1 / C2 / C3 contributions.

Ablation configs (paper Table IV):
  A: Baseline GICP      — alpha=0,   adaptive=False, window=1
  B: +C1                — alpha=0,   adaptive=True,  window=1
  C: +C2                — alpha=X,   adaptive=False, window=1
  D: C1+C2 (no C3)      — alpha=X,   adaptive=True,  window=1
  E: Full IV-GICP       — alpha=X,   adaptive=True,  window=10

Supported datasets:
  kitti   — KITTI seq00, 300fr (outdoor driving)
  subt    — SubT Final_UGV1, 300fr (mine/tunnel)
  metro   — GEODE Metro Shield_tunnel1, 300fr (metro tunnel)
  all     — run all three sequentially

Usage:
  python examples/run_ablation.py --dataset kitti --device cuda
  python examples/run_ablation.py --dataset all   --device cuda
  python examples/run_ablation.py --dataset subt  --max-frames 300
"""

import argparse
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from iv_gicp.pipeline import IVGICPPipeline

# ── Dataset paths ──────────────────────────────────────────────────────────────

KITTI_ROOT = Path("/home/km/data/kitti/dataset")
SUBT_ROOT  = Path("/home/km/data/SubT-MRS")
GEODE_ROOT = Path("/home/km/data/GEODE")


# ── ATE metric ────────────────────────────────────────────────────────────────

def ate_rmse(poses_est, poses_gt):
    n = min(len(poses_est), len(poses_gt))
    if n < 2:
        return float("nan")
    t_est = np.array([p[:3, 3] for p in poses_est[:n]])
    t_gt  = np.array([p[:3, 3] for p in poses_gt[:n]])
    mu_e  = t_est.mean(0); mu_g = t_gt.mean(0)
    H     = (t_est - mu_e).T @ (t_gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ D @ U.T
    t = mu_g - R @ mu_e
    errs = np.array([np.linalg.norm(R @ p[:3,3] + t - g[:3,3])
                     for p, g in zip(poses_est[:n], poses_gt[:n])])
    return float(np.sqrt(np.mean(errs**2)))


# ── KITTI data loading ─────────────────────────────────────────────────────────

def load_kitti(max_frames=300):
    seq = "00"
    velo_dir  = KITTI_ROOT / "sequences" / seq / "velodyne"
    pose_file = KITTI_ROOT / "poses" / f"{seq}.txt"

    bins = sorted(velo_dir.glob("*.bin"))[:max_frames]
    frames = []
    for bf in bins:
        data = np.fromfile(str(bf), dtype=np.float32).reshape(-1, 4)
        r = np.linalg.norm(data[:, :3], axis=1)
        frames.append(data[(r > 0.5) & (r < 80.0)].astype(np.float64))

    poses_gt = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4); T[:3, :] = np.array(vals).reshape(3, 4)
            poses_gt.append(T)
    poses_gt = poses_gt[:max_frames]
    return frames, poses_gt


# ── SubT data loading ─────────────────────────────────────────────────────────

_VLP16_ELEVATIONS = np.deg2rad(
    [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
)

def _decode_vlp16(packets_data_list):
    all_pts = []
    for data in packets_data_list:
        if len(data) < 1206:
            continue
        for block_id in range(12):
            offset = block_id * 100
            flag = struct.unpack_from('<H', data, offset)[0]
            if flag != 0xEEFF:
                continue
            azimuth_raw = struct.unpack_from('<H', data, offset + 2)[0]
            az_deg = azimuth_raw * 0.01
            if block_id < 11:
                next_offset = (block_id + 1) * 100
                if struct.unpack_from('<H', data, next_offset)[0] == 0xEEFF:
                    next_az = struct.unpack_from('<H', data, next_offset + 2)[0] * 0.01
                    az_step = ((next_az - az_deg) % 360.0) / 2.0
                else:
                    az_step = 0.18
            else:
                az_step = 0.18
            for seq in range(2):
                az_rad = np.deg2rad(az_deg + seq * az_step)
                ch_offset = offset + 4 + seq * 48
                cos_az = np.cos(az_rad); sin_az = np.sin(az_rad)
                raw = np.frombuffer(data[ch_offset:ch_offset+48], dtype=np.uint8)
                dist_raw = raw[0::3].astype(np.uint16) | (raw[1::3].astype(np.uint16) << 8)
                intensity = raw[2::3].astype(np.float32)
                valid = dist_raw > 0
                if not np.any(valid):
                    continue
                dist = dist_raw[valid] * 0.002
                cos_el = np.cos(_VLP16_ELEVATIONS[valid])
                sin_el = np.sin(_VLP16_ELEVATIONS[valid])
                x = dist * cos_el * sin_az
                y = dist * cos_el * cos_az
                z = dist * sin_el
                all_pts.append(np.column_stack([x, y, z, intensity[valid]]))
    return np.vstack(all_pts).astype(np.float64) if all_pts else np.zeros((0, 4))


def load_subt(max_frames=300):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_types_from_msg, get_typestore, Stores

    ds_info = {
        "rosbag": "rosbag/SubT_MRS_Final_Challenge_UGV1.zip",
        "gt":     "LiDAR_Inertial_Track/SubT_MRS_Final_Challenge_UGV1.zip",
        "gt_file": "SubT_MRS_Final_Challenge_UGV1/ground_truth_path.csv",
    }

    # Load GT
    gt_zip = SUBT_ROOT / ds_info["gt"]
    with zipfile.ZipFile(gt_zip) as z:
        with z.open(ds_info["gt_file"]) as f:
            lines = f.read().decode().strip().split('\n')
    gt_ts, gt_poses = [], []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue
        ts = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        T = np.eye(4)
        T[0,0]=1-2*(qy**2+qz**2); T[0,1]=2*(qx*qy-qw*qz); T[0,2]=2*(qx*qz+qw*qy)
        T[1,0]=2*(qx*qy+qw*qz);   T[1,1]=1-2*(qx**2+qz**2); T[1,2]=2*(qy*qz-qw*qx)
        T[2,0]=2*(qx*qz-qw*qy);   T[2,1]=2*(qy*qz+qw*qx);   T[2,2]=1-2*(qx**2+qy**2)
        T[:3,3] = [x, y, z]
        gt_ts.append(ts); gt_poses.append(T)
    gt_ts = np.array(gt_ts, dtype=np.int64)
    T0_inv = np.linalg.inv(gt_poses[0])
    gt_poses = [T0_inv @ p for p in gt_poses]

    # Load LiDAR frames
    bag_zip = SUBT_ROOT / ds_info["rosbag"]
    import tempfile, os as _os

    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}
    for t, msgdef in [
        ('velodyne_msgs/msg/VelodynePacket', 'time stamp\nuint8[1206] data\n'),
        ('velodyne_msgs/msg/VelodyneScan',
         'std_msgs/Header header\nvelodyne_msgs/VelodynePacket[] packets\n'),
    ]:
        add_types.update(get_types_from_msg(msgdef, t))
    typestore.register(add_types)

    frames = []
    with zipfile.ZipFile(bag_zip) as z:
        bag_files = sorted([n for n in z.namelist() if n.endswith('.bag')])
        for bag_name in bag_files:
            bag_data = z.read(bag_name)
            with tempfile.NamedTemporaryFile(suffix='.bag', delete=False) as tf:
                tf.write(bag_data)
                tmp_path = tf.name
            try:
                with Reader(tmp_path) as bag:
                    pkt_conns = [c for c in bag.connections
                                 if 'velodyne' in c.topic.lower()
                                 and c.msgtype == 'velodyne_msgs/msg/VelodyneScan']
                    for conn, ts, raw in bag.messages(connections=pkt_conns):
                        msg = typestore.deserialize_ros1(raw, 'velodyne_msgs/msg/VelodyneScan')
                        pkts = [bytes(p.data) for p in msg.packets]
                        pts = _decode_vlp16(pkts)
                        r = np.linalg.norm(pts[:, :3], axis=1)
                        pts = pts[(r > 0.5) & (r < 80.0)]
                        if len(pts) > 100:
                            ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                            frames.append((ts_ns, pts))
                        if len(frames) % 50 == 0:
                            print(f"  Loaded {len(frames)}/{max_frames} ...", end="\r")
                        if len(frames) >= max_frames:
                            break
            finally:
                _os.unlink(tmp_path)
            if len(frames) >= max_frames:
                break

    print(f"\n  Done: {len(frames)} frames")

    # Match GT to frames
    frame_ts = np.array([f[0] for f in frames], dtype=np.int64)
    matched_gt = []
    for ts in frame_ts:
        idx = np.searchsorted(gt_ts, ts)
        idx = np.clip(idx, 0, len(gt_ts)-1)
        matched_gt.append(gt_poses[idx])

    cloud_frames = [f[1] for f in frames]
    return cloud_frames, matched_gt


# ── GEODE Metro data loading ───────────────────────────────────────────────────

def _parse_livox(raw: bytes, max_range=60.0):
    data = bytes(raw)
    secs, nsecs = struct.unpack_from('<II', data, 4)
    fid_len = struct.unpack_from('<I', data, 12)[0]
    off = 16 + fid_len + 8 + 4 + 1 + 3
    arr_len = struct.unpack_from('<I', data, off)[0]; off += 4
    arr = np.frombuffer(data[off:off+19*arr_len], dtype=np.uint8).reshape(arr_len, 19)
    x = arr[:,4:8].view(np.float32).reshape(-1)
    y = arr[:,8:12].view(np.float32).reshape(-1)
    z = arr[:,12:16].view(np.float32).reshape(-1)
    refl = arr[:,16].astype(np.float64) / 255.0
    pts = np.stack([x, y, z, refl], axis=1).astype(np.float64)
    r = np.linalg.norm(pts[:,:3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    return secs*1e9+nsecs, pts[valid]


def load_metro(max_frames=300):
    from rosbags.rosbag1 import Reader
    from scipy.spatial.transform import Rotation

    bag_path = GEODE_ROOT / "sensor_data/Metro_tunnel/Shield_tunnel1_gamma/Shield_tunnel1_gamma.bag"
    gt_path  = GEODE_ROOT / "groundtruth/metro_tunnel/Shield_tunnel1.txt"

    # GT
    data = np.loadtxt(gt_path)
    gt_ts   = data[:, 0]
    gt_txyz = data[:, 1:4]
    gt_quat = data[:, 4:8]
    T0_inv  = None
    gt_poses = []
    for i in range(len(gt_ts)):
        q = gt_quat[i]
        R = Rotation.from_quat(q).as_matrix() if np.linalg.norm(q) > 1e-6 else np.eye(3)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = gt_txyz[i]
        if T0_inv is None:
            T0_inv = np.linalg.inv(T)
        gt_poses.append(T0_inv @ T)
    gt_ts    = np.array(gt_ts)
    gt_poses = np.stack(gt_poses)

    # Frames
    frames = []
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == '/livox/lidar']
        for conn, ts_ns, raw in bag.messages(connections=conns):
            try:
                t_ns, pts = _parse_livox(raw)
                if pts is not None and len(pts) > 100:
                    frames.append((t_ns/1e9, pts))
                    if len(frames) >= max_frames:
                        break
            except Exception:
                continue

    print(f"  Done: {len(frames)} frames")

    # Match GT
    frame_ts = np.array([f[0] for f in frames])
    matched_gt = []
    for t in frame_ts:
        idx = int(np.searchsorted(gt_ts, t))
        idx = max(0, min(idx, len(gt_ts)-1))
        matched_gt.append(gt_poses[idx])

    return [f[1] for f in frames], matched_gt


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_config(label, frames, poses_gt, alpha, adaptive, window, params, device):
    pipeline = IVGICPPipeline(
        alpha=alpha,
        adaptive_voxelization=adaptive,
        window_size=window,
        device=device,
        **params,
    )
    abs_poses = []
    times = []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        times.append((time.perf_counter() - t0) * 1000)
        abs_poses.append(result.pose.copy())
        if i % 50 == 0 or i == len(frames)-1:
            print(f"  {i:4d}/{len(frames)}  {times[-1]:6.1f}ms", end="\r")
    print()
    ate = ate_rmse(abs_poses, poses_gt)
    hz  = 1000.0 / np.mean(times[1:]) if len(times) > 1 else 0.0
    print(f"  [{label}]  ATE={ate:.3f}m  {hz:.1f}Hz")
    return ate, hz


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = {
    "kitti": {
        "loader": load_kitti,
        "loader_kw": {},
        "alpha": 0.1,
        "params": dict(
            voxel_size=1.0, source_voxel_size=0.3,
            max_correspondence_distance=2.0, initial_threshold=2.0,
            max_map_frames=10, max_iterations=30,
            min_range=0.5, max_range=80.0,
        ),
        "label": "KITTI seq00 (outdoor)",
    },
    "subt": {
        "loader": load_subt,
        "loader_kw": {},
        "alpha": 0.1,
        "params": dict(
            voxel_size=0.5, source_voxel_size=0.3,
            max_correspondence_distance=2.0, initial_threshold=1.5,
            max_map_frames=30, max_iterations=30,
            min_range=0.3, max_range=80.0,
        ),
        "label": "SubT Final_UGV1 (mine/tunnel)",
    },
    "metro": {
        "loader": load_metro,
        "loader_kw": {},
        "alpha": 0.5,
        "params": dict(
            voxel_size=0.3, source_voxel_size=0.2,
            max_correspondence_distance=0.5, initial_threshold=0.5,
            max_map_frames=30, max_iterations=20,
            min_range=0.5, max_range=60.0,
        ),
        "label": "GEODE Metro (metro tunnel)",
    },
}

ABLATION_CONFIGS = [
    # (label,       C1_adaptive, C2_alpha_ratio, C3_window)
    ("A: GICP-Base",   False, 0.0, 1),
    ("B: +C1",         True,  0.0, 1),
    ("C: +C2",         False, 1.0, 1),
    ("D: C1+C2",       True,  1.0, 1),
    ("E: Full(C1+2+3)", True,  1.0, 10),
]


def run_dataset(ds_key, max_frames, device):
    ds = DATASETS[ds_key]
    print(f"\n{'='*70}")
    print(f"  Dataset: {ds['label']}  ({max_frames} frames)")
    print(f"{'='*70}")

    print("Loading data...")
    frames, poses_gt = ds["loader"](max_frames=max_frames, **ds["loader_kw"])
    print(f"  {len(frames)} frames, {len(poses_gt)} GT poses")

    results = []
    for cfg_label, adaptive, alpha_ratio, window in ABLATION_CONFIGS:
        alpha = ds["alpha"] * alpha_ratio
        print(f"\n[{cfg_label}]  adaptive={adaptive}  α={alpha:.2f}  window={window}")
        ate, hz = run_config(
            cfg_label, frames, poses_gt,
            alpha=alpha, adaptive=adaptive, window=window,
            params=ds["params"], device=device,
        )
        results.append((cfg_label, adaptive, alpha_ratio > 0, window > 1, ate, hz))

    # Summary table
    print(f"\n{'─'*70}")
    print(f"  {ds['label']}")
    print(f"{'─'*70}")
    print(f"  {'Config':<22}  C1  C2  C3    ATE(m)   Hz")
    print(f"  {'─'*60}")
    best_ate = min(r[4] for r in results if not np.isnan(r[4]))
    for cfg_label, ada, inten, win, ate, hz in results:
        c1 = "Y" if ada   else "-"
        c2 = "Y" if inten else "-"
        c3 = "Y" if win   else "-"
        ate_str = f"{ate:.3f}m"
        marker  = " *" if abs(ate - best_ate) < 1e-6 else "  "
        print(f"  {cfg_label:<22}  {c1}   {c2}   {c3}  {ate_str:>8} {hz:5.1f}{marker}")
    print(f"{'─'*70}")
    return results


def main():
    parser = argparse.ArgumentParser(description="IV-GICP Ablation Study")
    parser.add_argument("--dataset", choices=["kitti", "subt", "metro", "all"],
                        default="all")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_results = {}
    for k in keys:
        all_results[k] = run_dataset(k, args.max_frames, args.device)

    # Final combined summary
    if len(keys) > 1:
        print(f"\n{'='*70}")
        print("  ABLATION SUMMARY")
        print(f"{'='*70}")
        header = f"  {'Config':<22}  {'KITTI':>8}  {'SubT':>8}  {'Metro':>8}"
        print(header)
        print(f"  {'─'*60}")
        n = len(ABLATION_CONFIGS)
        for i in range(n):
            cfg_label = ABLATION_CONFIGS[i][0]
            row = f"  {cfg_label:<22}"
            for k in keys:
                ate = all_results[k][i][4]
                row += f"  {ate:>7.3f}m"
            print(row)
        print(f"{'='*70}")

    # Save JSON
    out_path = Path(__file__).parent.parent / "results" / "ablation_results.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = {k: [(r[0], r[4], r[5]) for r in v] for k, v in all_results.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
