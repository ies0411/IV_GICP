"""
GEODE Metro Tunnel Evaluation Script
=====================================
Evaluates IV-GICP vs GICP-Baseline vs KISS-ICP on GEODE Metro Tunnel dataset.
Sensor: Livox Mid-360 (non-repetitive scan, 24000 pts/frame at 10Hz)
Topic: /livox/lidar  (livox_ros_driver/msg/CustomMsg, 19 bytes/point)

CustomPoint layout (19 bytes):
  offset_time (uint32, 4B)
  x (float32, 4B), y (float32, 4B), z (float32, 4B)
  reflectivity (uint8, 1B), tag (uint8, 1B), line (uint8, 1B)

GT: timestamp x_utm y_utm z qx qy qz qw

Usage:
    uv run python examples/run_metro_eval.py --seq 1 --max-frames 300
    uv run python examples/run_metro_eval.py --seq 2 --device cuda
"""

import argparse
import json
import struct
import time
from pathlib import Path
import numpy as np

GEODE_ROOT = Path("/home/km/data/GEODE")
LIDAR_TOPIC = "/livox/lidar"


# ── Livox frame parser ────────────────────────────────────────────────────────

def parse_livox_frame(raw: bytes, max_range: float = 60.0):
    """
    Parse livox_ros_driver/msg/CustomMsg from raw ROS1 bytes.
    Returns (timestamp_ns, points) where points is (N,4) float64 [x,y,z,reflectivity_norm].
    """
    data = bytes(raw)
    secs, nsecs = struct.unpack_from('<II', data, 4)
    fid_len = struct.unpack_from('<I', data, 12)[0]
    off = 16 + fid_len
    # skip timebase(u64) + point_num(u32) + lidar_id(u8) + rsvd[3]
    off += 8 + 4 + 1 + 3
    arr_len = struct.unpack_from('<I', data, off)[0]
    off += 4
    # Each point: offset_time(4) x(4) y(4) z(4) reflectivity(1) tag(1) line(1) = 19 bytes
    arr = np.frombuffer(data[off:off + 19 * arr_len], dtype=np.uint8).reshape(arr_len, 19)
    x    = arr[:,  4: 8].view(np.float32).reshape(-1)
    y    = arr[:,  8:12].view(np.float32).reshape(-1)
    z    = arr[:, 12:16].view(np.float32).reshape(-1)
    refl = arr[:, 16].astype(np.float64) / 255.0  # normalize to [0,1]

    pts = np.stack([x, y, z, refl], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    ts_ns = secs * 1e9 + nsecs
    return ts_ns, pts[valid]


def read_metro_frames(bag_path: Path, max_frames: int = None, max_range: float = 60.0):
    from rosbags.rosbag1 import Reader

    frames = []
    print(f"Reading: {bag_path}")
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = conns[0].msgcount if conns else 0
        print(f"  Livox frames in bag: {total}")
        for conn, ts_ns, raw in bag.messages(connections=conns):
            try:
                t_ns, pts = parse_livox_frame(raw, max_range=max_range)
                if pts is not None and len(pts) > 100:
                    frames.append((t_ns / 1e9, pts))
                    if len(frames) % 200 == 0:
                        print(f"  Loaded {len(frames)}/{total} ...", end="\r")
                    if max_frames and len(frames) >= max_frames:
                        break
            except Exception as e:
                continue
    if not frames:
        raise RuntimeError(f"No frames loaded from {bag_path}")
    print(f"\n  Done: {len(frames)} frames  "
          f"({frames[-1][0]-frames[0][0]:.1f}s, "
          f"avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame)")
    return frames


# ── GT loader (same format as Urban tunnel) ───────────────────────────────────

def load_gt(gt_path: Path):
    """Load GT: timestamp x_utm y_utm z qx qy qz qw → local SE(3) poses."""
    from scipy.spatial.transform import Rotation

    data = np.loadtxt(gt_path)
    timestamps = data[:, 0]
    txyz  = data[:, 1:4]
    quats = data[:, 4:8]  # qx qy qz qw

    T0_inv = None
    poses  = []
    for i in range(len(timestamps)):
        R = Rotation.from_quat(quats[i]).as_matrix()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = txyz[i]
        if T0_inv is None:
            T0_inv = np.linalg.inv(T)
        poses.append(T0_inv @ T)
    return timestamps, np.stack(poses)


def interpolate_gt(gt_times, gt_poses, lidar_times):
    idx  = np.searchsorted(gt_times, lidar_times)
    idx  = np.clip(idx, 0, len(gt_times) - 1)
    prev = np.clip(idx - 1, 0, len(gt_times) - 1)
    dt_b = np.abs(gt_times[prev] - lidar_times)
    dt_a = np.abs(gt_times[idx]  - lidar_times)
    best = np.where(dt_b < dt_a, prev, idx)
    return gt_poses[best]


def save_tum(poses, timestamps, path: Path):
    from scipy.spatial.transform import Rotation
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def traj_stats(poses):
    dists = [np.linalg.norm(poses[i+1][:3, 3] - poses[i][:3, 3])
             for i in range(len(poses) - 1)]
    return float(np.sum(dists)), float(np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]))


def compose_poses(rel_list):
    poses = [np.eye(4)]
    for T in rel_list:
        poses.append(poses[-1] @ T)
    return poses


# ── Method runners (same as run_geode_eval.py) ────────────────────────────────

def run_iv_gicp(frames, device="cuda"):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(
        voxel_size=0.5,
        source_voxel_size=0.2,
        alpha=0.5,
        max_correspondence_distance=1.5,
        initial_threshold=1.5,
        max_iterations=20,
        device=device,
        use_distribution_propagation=False,
    )
    abs_poses, times = [np.eye(4)], []
    print(f"\n[IV-GICP] {len(frames)} frames  device={device}  α=0.5")
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy()); times.append(elapsed)
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:6.0f}ms", end="\r")
    _print_timing("IV-GICP", times[1:])
    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    return rel, {"total": times[1:]}


def run_gicp_baseline(frames, device="cuda"):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(
        voxel_size=0.5,
        source_voxel_size=0.2,
        alpha=0.0,
        max_correspondence_distance=1.5,
        initial_threshold=1.5,
        max_iterations=20,
        device=device,
        use_distribution_propagation=False,
    )
    abs_poses, times = [np.eye(4)], []
    print(f"\n[GICP-Baseline] {len(frames)} frames  device={device}  α=0.0")
    for i, (ts, pts) in enumerate(frames):
        pts_g = pts.copy(); pts_g[:, 3] = 0.0
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts_g[:, :3], pts_g[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy()); times.append(elapsed)
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:6.0f}ms", end="\r")
    _print_timing("GICP-Baseline", times[1:])
    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    return rel, {"total": times[1:]}


def run_kiss_icp(frames):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    config = KISSConfig()
    config.data.deskew   = False
    config.data.max_range = 60.0
    config.data.min_range = 0.5
    config.mapping.voxel_size = 0.8

    od = KissICP(config=config)
    abs_poses, times = [np.eye(4)], []
    print(f"\n[KISS-ICP] {len(frames)} frames")
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(od.last_pose.copy()); times.append(elapsed)
        if i % 200 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:5.1f}ms", end="\r")
    _print_timing("KISS-ICP", times)
    return abs_poses, {"total": times}


def _print_timing(name, times):
    t = np.array(times)
    print(f"\n  [{name}] mean={t.mean():.1f}ms  median={np.median(t):.1f}ms"
          f"  std={t.std():.1f}ms  → {1000/t.mean():.2f} Hz")


def compute_ate(gt_tum: Path, pred_tum: Path):
    try:
        from evo.tools import file_interface
        from evo.core import sync, metrics
        ref = file_interface.read_tum_trajectory_file(str(gt_tum))
        est = file_interface.read_tum_trajectory_file(str(pred_tum))
        ref, est = sync.associate_trajectories(ref, est)
        est.align(ref, correct_scale=False)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, est))
        return float(ape.get_statistic(metrics.StatisticsType.rmse))
    except Exception as e:
        print(f"  evo ATE failed: {e}"); return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="1", choices=["1","2","3"],
                        help="Shield tunnel sequence number (1/2/3)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device",    default="auto")
    parser.add_argument("--kiss-only", action="store_true")
    parser.add_argument("--no-gicp-baseline", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    seq_bag = f"Shield_tunnel{args.seq}_gamma"
    seq_gt  = f"Shield_tunnel{args.seq}"
    bag_path = GEODE_ROOT / "sensor_data"  / "Metro_tunnel" / seq_bag / f"{seq_bag}.bag"
    gt_path  = GEODE_ROOT / "groundtruth"  / "metro_tunnel" / f"{seq_gt}.txt"
    out = Path(__file__).parent.parent / "results" / "geode" / seq_bag
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    frames = read_metro_frames(bag_path, max_frames=args.max_frames)
    n = len(frames)
    duration  = frames[-1][0] - frames[0][0]
    timestamps = [f[0] for f in frames]

    # ── GT ────────────────────────────────────────────────────────────────────
    print(f"\nLoading GT: {gt_path}")
    gt_times, gt_poses = load_gt(gt_path)
    gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
    save_tum(gt_at_lidar, timestamps, out / "gt.tum")
    gt_path_len, gt_end_disp = traj_stats(gt_at_lidar)
    print(f"  GT: path={gt_path_len:.1f}m  end_disp={gt_end_disp:.1f}m")

    results = {}

    # ── KISS-ICP ──────────────────────────────────────────────────────────────
    ki_poses, ki_t = run_kiss_icp(frames)
    save_tum(ki_poses, timestamps, out / "kiss_icp.tum")
    pl, ed = traj_stats(ki_poses)
    ate = compute_ate(out / "gt.tum", out / "kiss_icp.tum")
    results["KISS-ICP"] = {"mean_ms": float(np.mean(ki_t["total"])),
                           "hz": float(1000/np.mean(ki_t["total"])),
                           "path_length_m": pl, "end_displacement_m": ed,
                           "ate_rmse_m": ate, "n_frames": n}

    if not args.kiss_only:
        # ── IV-GICP ───────────────────────────────────────────────────────────
        iv_rel, iv_t = run_iv_gicp(frames, device=device)
        iv_poses = compose_poses(iv_rel)
        save_tum(iv_poses, timestamps, out / "iv_gicp.tum")
        pl, ed = traj_stats(iv_poses)
        ate = compute_ate(out / "gt.tum", out / "iv_gicp.tum")
        results["IV-GICP"] = {"mean_ms": float(np.mean(iv_t["total"])),
                              "hz": float(1000/np.mean(iv_t["total"])),
                              "path_length_m": pl, "end_displacement_m": ed,
                              "ate_rmse_m": ate, "n_frames": n, "device": device}

        if not args.no_gicp_baseline:
            gb_rel, gb_t = run_gicp_baseline(frames, device=device)
            gb_poses = compose_poses(gb_rel)
            save_tum(gb_poses, timestamps, out / "gicp_baseline.tum")
            pl, ed = traj_stats(gb_poses)
            ate = compute_ate(out / "gt.tum", out / "gicp_baseline.tum")
            results["GICP-Baseline"] = {"mean_ms": float(np.mean(gb_t["total"])),
                                        "hz": float(1000/np.mean(gb_t["total"])),
                                        "path_length_m": pl, "end_displacement_m": ed,
                                        "ate_rmse_m": ate, "n_frames": n, "device": device}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  GEODE {seq_bag}  |  {n} frames  |  {duration:.1f}s")
    print(f"  GT:  path={gt_path_len:.1f}m  end_disp={gt_end_disp:.2f}m")
    print(f"{'='*76}")
    print(f"  {'Method':<22} {'ms/f':>6} {'Hz':>5} {'Path(m)':>9} "
          f"{'End(m)':>8} {'ATE RMSE':>10} {'Dev':>5}")
    print(f"  {'-'*72}")
    for name, r in results.items():
        ate_s = f"{r['ate_rmse_m']:.3f}m" if r.get("ate_rmse_m") else "   N/A"
        print(f"  {name:<22} {r['mean_ms']:>6.1f} {r['hz']:>5.1f} "
              f"{r['path_length_m']:>9.1f} {r['end_displacement_m']:>8.2f} "
              f"{ate_s:>10} {r.get('device','cpu'):>5}")
    print(f"{'='*76}")

    meta = {"seq": seq_bag, "bag": str(bag_path), "n_frames": n,
            "duration_s": duration,
            "avg_pts_per_frame": float(np.mean([len(f[1]) for f in frames])),
            "gt_path_m": gt_path_len, "gt_end_disp_m": gt_end_disp}
    with open(out / "results.json", "w") as f:
        json.dump({"meta": meta, "methods": results}, f, indent=2)
    print(f"\n  Results → {out}/results.json\n")


if __name__ == "__main__":
    main()
