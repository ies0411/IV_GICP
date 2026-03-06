"""
SubT-MRS LiDAR Odometry Evaluation

Evaluates IV-GICP, GICP-Baseline, and KISS-ICP on SubT-MRS datasets.
Supports:
  - Urban Challenge UGV1/UGV2 (VLP-16, underground urban corridors)
  - Final Challenge UGV1/UGV2 (VLP-16, mine/tunnel)
  - Laurel Caverns Handheld3 (cave)

Data source: /home/km/data/SubT-MRS/
  rosbag/{dataset}.zip   → raw VLP-16 bags (velodyne_packets)
  LiDAR_Inertial_Track/{dataset}.zip → ground_truth_path.csv

Usage:
  python examples/run_subt_eval.py --dataset Urban_UGV1 --max-frames 500
  python examples/run_subt_eval.py --dataset Urban_UGV1 --all
"""
import sys
import os
import time
import zipfile
import argparse
import io
import tempfile
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    _KISS_AVAILABLE = True
except ImportError:
    _KISS_AVAILABLE = False

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_types_from_msg, get_typestore, Stores
    _ROSBAGS_AVAILABLE = True
except ImportError:
    _ROSBAGS_AVAILABLE = False

from iv_gicp.pipeline import IVGICPPipeline

# ─── SubT-MRS dataset registry ────────────────────────────────────────────────

BASE_DIR = "/home/km/data/SubT-MRS"

DATASETS = {
    "Urban_UGV1": {
        "rosbag": "rosbag/SubT_MRS_Urban_Challenge_UGV1.zip",
        "gt":     "LiDAR_Inertial_Track/SubT_MRS_Urban_Challenge_UGV1.zip",
        "gt_file": "SubT_MRS_Urban_Challenge_UGV1/ground_truth_path.csv",
        "sensor": "vlp16",
        "env": "underground urban corridor",
    },
    "Urban_UGV2": {
        "rosbag": "rosbag/SubT_MRS_Urban_Challenge_UGV2.zip",
        "gt":     "LiDAR_Inertial_Track/SubT_MRS_Urban_Challenge_UGV2.zip",
        "gt_file": "SubT_MRS_Urban_Challenge_UGV2/ground_truth_path.csv",
        "sensor": "vlp16",
        "env": "underground urban corridor (long)",
    },
    "Final_UGV1": {
        "rosbag": "rosbag/SubT_MRS_Final_Challenge_UGV1.zip",
        "gt":     "LiDAR_Inertial_Track/SubT_MRS_Final_Challenge_UGV1.zip",
        "gt_file": "SubT_MRS_Final_Challenge_UGV1/ground_truth_path.csv",
        "sensor": "vlp16",
        "env": "mine/tunnel challenge",
    },
    "Laurel": {
        "rosbag": "rosbag/SubT_MRS_Laurel_Caverns_Handheld3.zip",
        "gt":     "LiDAR_Inertial_Track/SubT_MRS_Laurel_Caverns_Handheld3.zip",
        "gt_file": "SubT_MRS_Laurel_Caverns_Handheld3/ground_truth_path.csv",
        "sensor": "vlp16",
        "env": "cave (Laurel Caverns)",
    },
}

# ─── VLP-16 packet decoder ───────────────────────────────────────────────────

# VLP-16 calibration: elevation angles (degrees) for each of the 16 channels
_VLP16_ELEVATIONS = np.deg2rad(
    [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
)
# Azimuth offset per firing (approximate 0.18° between the two firings per block)
_VLP16_FIRING_AZ_OFFSET = 0.18  # degrees

def decode_vlp16_scan(packets_data_list):
    """
    Decode a list of VLP-16 raw packet bytes (1206 bytes each) into a point cloud.

    Returns: np.ndarray (N, 4) = [x, y, z, intensity]
    """
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
            az_deg = azimuth_raw * 0.01  # degrees

            # Compute next-block azimuth for interpolation
            if block_id < 11:
                next_offset = (block_id + 1) * 100
                next_flag = struct.unpack_from('<H', data, next_offset)[0]
                if next_flag == 0xEEFF:
                    next_az = struct.unpack_from('<H', data, next_offset + 2)[0] * 0.01
                    # Handle wrap-around
                    diff = (next_az - az_deg) % 360.0
                    az_step = diff / 2.0
                else:
                    az_step = _VLP16_FIRING_AZ_OFFSET
            else:
                az_step = _VLP16_FIRING_AZ_OFFSET

            for seq in range(2):  # two firings per block
                az_seq = az_deg + seq * az_step
                az_rad = np.deg2rad(az_seq)
                ch_offset = offset + 4 + seq * 48
                cos_az = np.cos(az_rad)
                sin_az = np.sin(az_rad)
                cos_el = np.cos(_VLP16_ELEVATIONS)
                sin_el = np.sin(_VLP16_ELEVATIONS)

                # Read 16 channels at once
                raw = np.frombuffer(data[ch_offset:ch_offset + 48], dtype=np.uint8)
                dist_raw = raw[0::3].astype(np.uint16) | (raw[1::3].astype(np.uint16) << 8)
                intensity = raw[2::3].astype(np.float32)
                valid = dist_raw > 0
                if not np.any(valid):
                    continue

                dist = dist_raw[valid] * 0.002  # meters
                x = dist * cos_el[valid] * sin_az
                y = dist * cos_el[valid] * cos_az
                z = dist * sin_el[valid]
                pts = np.column_stack([x, y, z, intensity[valid]])
                all_pts.append(pts)

    if not all_pts:
        return np.zeros((0, 4))
    return np.vstack(all_pts).astype(np.float64)


# ─── GT loading ──────────────────────────────────────────────────────────────

def load_gt_csv(zip_path, csv_inner_path):
    """
    Load SubT-MRS GT CSV: timestamp, x, y, z, qx, qy, qz, qw
    Returns: (timestamps_ns: np.ndarray, poses: list of 4x4)
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_inner_path) as f:
            lines = f.read().decode().strip().split('\n')

    timestamps = []
    poses = []
    for line in lines[1:]:  # skip header
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue
        ts = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

        # Quaternion → rotation matrix
        T = np.eye(4)
        T[0, 0] = 1 - 2*(qy**2 + qz**2)
        T[0, 1] = 2*(qx*qy - qw*qz)
        T[0, 2] = 2*(qx*qz + qw*qy)
        T[1, 0] = 2*(qx*qy + qw*qz)
        T[1, 1] = 1 - 2*(qx**2 + qz**2)
        T[1, 2] = 2*(qy*qz - qw*qx)
        T[2, 0] = 2*(qx*qz - qw*qy)
        T[2, 1] = 2*(qy*qz + qw*qx)
        T[2, 2] = 1 - 2*(qx**2 + qy**2)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        timestamps.append(ts)
        poses.append(T)

    return np.array(timestamps, dtype=np.int64), poses


def load_frames_from_zipped_bags(rosbag_zip_path, max_frames=None):
    """
    Extract all LiDAR frames from a zip of ROS1 bags.
    Yields (timestamp_ns, point_cloud_4xN) tuples.
    """
    if not _ROSBAGS_AVAILABLE:
        raise RuntimeError("rosbags package not available. pip install rosbags")

    # Register velodyne message types
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}
    for t, msgdef in [
        ('velodyne_msgs/msg/VelodynePacket', 'time stamp\nuint8[1206] data\n'),
        ('velodyne_msgs/msg/VelodyneScan',
         'std_msgs/Header header\nvelodyne_msgs/VelodynePacket[] packets\n'),
    ]:
        add_types.update(get_types_from_msg(msgdef, t))
    typestore.register(add_types)

    n_frames = 0
    with zipfile.ZipFile(rosbag_zip_path, 'r') as z:
        bag_names = sorted([n for n in z.namelist() if n.endswith('.bag')])
        for bag_name in bag_names:
            # Extract bag to temp file (rosbags needs seekable file)
            bag_data = z.read(bag_name)
            with tempfile.NamedTemporaryFile(suffix='.bag', delete=False) as tf:
                tf.write(bag_data)
                tmp_path = tf.name

            try:
                with Reader(tmp_path) as reader:
                    conns = [c for c in reader.connections
                             if 'velodyne' in c.topic.lower() and c.msgtype == 'velodyne_msgs/msg/VelodyneScan']
                    if not conns:
                        continue
                    for _, ts_ns, rawdata in reader.messages(connections=conns):
                        msg = typestore.deserialize_ros1(rawdata, 'velodyne_msgs/msg/VelodyneScan')
                        packets_data = [bytes(pkt.data) for pkt in msg.packets]
                        cloud = decode_vlp16_scan(packets_data)
                        if len(cloud) < 100:
                            continue
                        yield ts_ns, cloud
                        n_frames += 1
                        if max_frames is not None and n_frames >= max_frames:
                            return
            finally:
                os.unlink(tmp_path)


# ─── ATE computation ─────────────────────────────────────────────────────────

def compute_ate(est_poses, gt_timestamps_ns, gt_poses, est_timestamps_ns):
    """
    Compute ATE RMSE (Umeyama alignment) between estimated and GT trajectories.
    Matches poses by nearest timestamp.
    """
    from scipy.spatial.transform import Rotation

    # Match estimated frames to GT by nearest timestamp
    est_ts = np.array(est_timestamps_ns)
    gt_ts = gt_timestamps_ns

    matched_est = []
    matched_gt = []
    for i, ts in enumerate(est_ts):
        idx = np.argmin(np.abs(gt_ts - ts))
        dt_ms = abs(gt_ts[idx] - ts) / 1e6
        if dt_ms < 500:  # within 500ms
            matched_est.append(est_poses[i][:3, 3])
            matched_gt.append(gt_poses[idx][:3, 3])

    if len(matched_est) < 5:
        return float('nan'), len(matched_est)

    P = np.array(matched_est).T   # (3, N)
    Q = np.array(matched_gt).T    # (3, N)

    # Umeyama alignment (translation only, rigid body)
    mu_p = P.mean(axis=1, keepdims=True)
    mu_q = Q.mean(axis=1, keepdims=True)
    Pc = P - mu_p
    Qc = Q - mu_q

    H = Pc @ Qc.T
    try:
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        t = mu_q - R @ mu_p
        P_aligned = R @ P + t
        errors = np.linalg.norm(P_aligned - Q, axis=0)
        ate_rmse = float(np.sqrt(np.mean(errors**2)))
    except np.linalg.LinAlgError:
        ate_rmse = float('nan')

    return ate_rmse, len(matched_est)


# ─── Run methods ─────────────────────────────────────────────────────────────

def run_iv_gicp(frames_gen, dataset_info, args, alpha=0.1):
    """Run IV-GICP on frames from generator."""
    pipeline = IVGICPPipeline(
        voxel_size=args.voxel_size,
        source_voxel_size=args.source_voxel,
        alpha=alpha,
        max_correspondence_distance=args.max_corr,
        initial_threshold=args.initial_threshold,
        max_iterations=args.max_iter,
        huber_delta=args.huber_delta,
        min_range=0.5,
        max_range=50.0,
        adaptive_voxelization=args.adaptive,
        max_map_frames=args.max_map_frames,
        window_size=args.window_size,
        device=args.device,
    )

    poses = []
    timestamps = []
    times_ms = []
    n = 0

    for ts_ns, cloud in frames_gen:
        t0 = time.perf_counter()
        result = pipeline.process_frame(
            cloud[:, :3],
            intensities=cloud[:, 3],
            timestamp=ts_ns / 1e9,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        poses.append(result.pose.copy())
        timestamps.append(ts_ns)
        times_ms.append(elapsed_ms)
        n += 1
        if n % 50 == 0:
            hz = 1000.0 / float(np.mean(times_ms[-50:])) if len(times_ms) >= 50 else 0
            print(f"  frame {n:4d}  {elapsed_ms:6.0f}ms  {hz:.1f}Hz", flush=True)

    return poses, timestamps, times_ms


def run_gicp_baseline(frames_gen, dataset_info, args):
    return run_iv_gicp(frames_gen, dataset_info, args, alpha=0.0)


def run_kiss(frames_gen, args):
    """Run KISS-ICP on frames."""
    if not _KISS_AVAILABLE:
        return None, None, None

    cfg = KISSConfig()
    cfg.data.deskew = False
    cfg.data.max_range = 80.0
    cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = args.voxel_size
    od = KissICP(config=cfg)

    poses = []
    timestamps = []
    times_ms = []

    for i, (ts_ns, cloud) in enumerate(frames_gen):
        t0 = time.perf_counter()
        src = cloud[:, :3].astype(np.float64)
        od.register_frame(src, np.full(len(src), float(i)))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        poses.append(od.last_pose.copy())
        timestamps.append(ts_ns)
        times_ms.append(elapsed_ms)

    return poses, timestamps, times_ms


# ─── Main ────────────────────────────────────────────────────────────────────

def evaluate_dataset(dataset_name, args):
    """Evaluate all methods on a single dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[dataset_name]
    rosbag_zip = os.path.join(BASE_DIR, info["rosbag"])
    gt_zip     = os.path.join(BASE_DIR, info["gt"])

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  ({info['env']})")
    print(f"{'='*60}")

    # Load GT
    gt_ts, gt_poses = load_gt_csv(gt_zip, info["gt_file"])
    print(f"GT: {len(gt_poses)} poses, duration {(gt_ts[-1]-gt_ts[0])/1e9:.1f}s")

    results = {}

    # ── IV-GICP ──────────────────────────────────────────────────────────────
    print(f"\n[IV-GICP] alpha={args.alpha}  max_frames={args.max_frames}")
    gen = load_frames_from_zipped_bags(rosbag_zip, args.max_frames)
    t_total = time.perf_counter()
    poses, timestamps, times = run_iv_gicp(gen, info, args, alpha=args.alpha)
    total_s = time.perf_counter() - t_total
    if poses:
        path_len = sum(np.linalg.norm(poses[i][:3,3] - poses[i-1][:3,3])
                       for i in range(1, len(poses)))
        ate, n_matched = compute_ate(poses, gt_ts, gt_poses, timestamps)
        avg_ms = float(np.mean(times[1:])) if len(times) > 1 else float(times[0])
        print(f"  ATE RMSE: {ate:.3f}m  path: {path_len:.1f}m  {1000/avg_ms:.1f}Hz  "
              f"({n_matched}/{len(poses)} matched)")
        results["IV-GICP"] = {"ate": ate, "path": path_len, "hz": 1000/avg_ms,
                               "n_frames": len(poses), "times": times}

    # ── GICP Baseline ─────────────────────────────────────────────────────────
    print(f"\n[GICP-Baseline] (alpha=0)")
    gen = load_frames_from_zipped_bags(rosbag_zip, args.max_frames)
    poses_b, timestamps_b, times_b = run_iv_gicp(gen, info, args, alpha=0.0)
    if poses_b:
        path_b = sum(np.linalg.norm(poses_b[i][:3,3] - poses_b[i-1][:3,3])
                     for i in range(1, len(poses_b)))
        ate_b, n_b = compute_ate(poses_b, gt_ts, gt_poses, timestamps_b)
        avg_b = float(np.mean(times_b[1:])) if len(times_b) > 1 else float(times_b[0])
        print(f"  ATE RMSE: {ate_b:.3f}m  path: {path_b:.1f}m  {1000/avg_b:.1f}Hz  "
              f"({n_b}/{len(poses_b)} matched)")
        results["GICP-Baseline"] = {"ate": ate_b, "path": path_b, "hz": 1000/avg_b,
                                     "n_frames": len(poses_b), "times": times_b}

    # ── KISS-ICP ─────────────────────────────────────────────────────────────
    if _KISS_AVAILABLE:
        print(f"\n[KISS-ICP]")
        gen = load_frames_from_zipped_bags(rosbag_zip, args.max_frames)
        poses_k, timestamps_k, times_k = run_kiss(gen, args)
        if poses_k:
            path_k = sum(np.linalg.norm(poses_k[i][:3,3] - poses_k[i-1][:3,3])
                         for i in range(1, len(poses_k)))
            ate_k, n_k = compute_ate(poses_k, gt_ts, gt_poses, timestamps_k)
            avg_k = float(np.mean(times_k[1:])) if len(times_k) > 1 else float(times_k[0])
            print(f"  ATE RMSE: {ate_k:.3f}m  path: {path_k:.1f}m  {1000/avg_k:.1f}Hz  "
                  f"({n_k}/{len(poses_k)} matched)")
            results["KISS-ICP"] = {"ate": ate_k, "path": path_k, "hz": 1000/avg_k,
                                    "n_frames": len(poses_k), "times": times_k}

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Method':<18} {'ATE(m)':>8} {'Path(m)':>8} {'Hz':>6} {'Frames':>7}")
    print(f"{'─'*60}")
    for method, r in results.items():
        ate_str = f"{r['ate']:.3f}" if not np.isnan(r.get('ate', float('nan'))) else "  n/a"
        print(f"{method:<18} {ate_str:>8} {r['path']:>8.1f} {r['hz']:>6.1f} {r['n_frames']:>7}")
    print(f"{'─'*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SubT-MRS LiDAR Odometry Evaluation")
    parser.add_argument("--dataset",   default="Urban_UGV1",
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Dataset to evaluate")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum frames to process (default: 300)")
    parser.add_argument("--voxel-size", type=float, default=0.5,
                        help="Target map voxel size (m)")
    parser.add_argument("--source-voxel", type=float, default=0.3,
                        help="Source downsampling voxel size (m)")
    parser.add_argument("--max-corr",  type=float, default=2.0,
                        help="Max correspondence distance (m)")
    parser.add_argument("--initial-threshold", type=float, default=1.5,
                        help="Initial adaptive threshold sigma")
    parser.add_argument("--max-iter",  type=int,   default=30,
                        help="Max GN iterations")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber kernel threshold (0=disabled)")
    parser.add_argument("--alpha",     type=float, default=0.1,
                        help="Intensity weight (0=GICP)")
    parser.add_argument("--adaptive",  action="store_true", default=False,
                        help="Use adaptive voxelization (C1)")
    parser.add_argument("--max-map-frames", type=int, default=30,
                        help="Sliding window size in frames")
    parser.add_argument("--device",      default="auto",
                        help="Device: auto/cuda/cpu")
    parser.add_argument("--window-size", type=int, default=1,
                        help="FORM window smoothing size (1=disabled, e.g. 10)")
    parser.add_argument("--skip-genz",   action="store_true")
    args = parser.parse_args()

    if args.dataset == "all":
        all_results = {}
        for ds in DATASETS:
            all_results[ds] = evaluate_dataset(ds, args)
        print("\n\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"{'Dataset':<18} {'Method':<18} {'ATE(m)':>8} {'Hz':>6}")
        print("─"*70)
        for ds, results in all_results.items():
            for method, r in results.items():
                ate_str = f"{r['ate']:.3f}" if not np.isnan(r.get('ate', float('nan'))) else "   n/a"
                print(f"{ds:<18} {method:<18} {ate_str:>8} {r['hz']:>6.1f}")
    else:
        evaluate_dataset(args.dataset, args)


if __name__ == "__main__":
    main()
