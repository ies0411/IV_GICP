"""
GEODE Urban Tunnel Evaluation Script
=====================================
Evaluates IV-GICP vs GICP-Baseline vs KISS-ICP on GEODE dataset
Urban Tunnel sequences (Urban_Tunnel01/02/03).

These sequences expose severe geometric degeneracy:
- Long urban tunnels: parallel walls + ceiling → geometry is near-singular
- Ground truth via RTK-GPS/INS → ATE directly measurable

Usage:
    uv run python examples/run_geode_eval.py                          # Tunnel01, all methods
    uv run python examples/run_geode_eval.py --seq 02                 # Tunnel02
    uv run python examples/run_geode_eval.py --max-frames 300         # fast subset test
    uv run python examples/run_geode_eval.py --kiss-only              # KISS-ICP only
    uv run python examples/run_geode_eval.py --no-gicp-baseline       # skip baseline

Output:
    results/geode/Urban_TunnelXX/iv_gicp.tum
    results/geode/Urban_TunnelXX/kiss_icp.tum
    results/geode/Urban_TunnelXX/gicp_baseline.tum
    results/geode/Urban_TunnelXX/gt.tum          (GT in local frame for evo_ape)
    results/geode/Urban_TunnelXX/results.json

ATE evaluation:
    evo_ape tum results/geode/Urban_TunnelXX/gt.tum \\
                results/geode/Urban_TunnelXX/iv_gicp.tum --align --plot
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

GEODE_ROOT = Path("/home/km/data/GEODE")
LIDAR_TOPIC = "/velodyne_points"


# ── GEODE bag reader ─────────────────────────────────────────────────────────

def read_geode_frames(bag_path: Path, max_frames: int = None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    frames = []
    print(f"Reading: {bag_path}")
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = conns[0].msgcount if conns else 0
        print(f"  LiDAR frames in bag: {total}")
        for conn, ts_ns, rawdata in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            pts = _parse_velodyne_pc2(msg)
            if pts is not None and len(pts) > 100:
                frames.append((ts_ns / 1e9, pts))
                if len(frames) % 200 == 0:
                    print(f"  Loaded {len(frames)}/{total} ...", end="\r")
                if max_frames and len(frames) >= max_frames:
                    break
    if not frames:
        raise RuntimeError(f"No frames loaded from {bag_path}")
    print(f"\n  Done: {len(frames)} frames  "
          f"({frames[-1][0]-frames[0][0]:.1f}s, "
          f"avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame)")
    return frames


def _parse_velodyne_pc2(msg, max_range: float = 25.0) -> np.ndarray:
    """
    Velodyne VLP-16 PointCloud2 → (N,4) float64 [x,y,z,intensity].
    Fields: x(f32,0) y(f32,4) z(f32,8) intensity(f32,12) ring(u16,16) time(f32,18)
    point_step = 22 bytes

    max_range: clip to nearby points only. In urban tunnel VLP-16 scans, most
    useful geometry (walls, ceiling, floor) is within 25m. Limiting range
    increases voxel point density: fewer voxels cover the visible volume →
    more points per voxel → better covariance estimates → better GICP.
    """
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 12:16].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    return pts[valid]


# ── Ground truth parsing ──────────────────────────────────────────────────────

def load_geode_gt(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse GEODE GT file: timestamp x_utm y_utm z qx qy qz qw
    Returns:
        timestamps: (N,) float64
        poses_SE3: (N,4,4) float64 — in LOCAL frame (T0^{-1} @ Ti)
    """
    from scipy.spatial.transform import Rotation

    data = np.loadtxt(gt_path)
    timestamps = data[:, 0]
    txyz = data[:, 1:4]         # UTM x, UTM y, altitude z
    quats = data[:, 4:8]        # qx qy qz qw

    # Build SE(3) poses in world frame (UTM)
    T0_inv = None
    poses = []
    for i in range(len(timestamps)):
        R = Rotation.from_quat(quats[i]).as_matrix()  # qx qy qz qw
        t = txyz[i]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        if T0_inv is None:
            T0_inv = np.linalg.inv(T)
        poses.append(T0_inv @ T)

    return timestamps, np.stack(poses)


def interpolate_gt_at_lidar_times(gt_times: np.ndarray, gt_poses: np.ndarray,
                                   lidar_times: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour interpolation: find GT pose closest to each LiDAR timestamp.
    Returns (M, 4, 4) array.
    """
    indices = np.searchsorted(gt_times, lidar_times)
    indices = np.clip(indices, 0, len(gt_times) - 1)
    # pick nearest (before or after)
    prev_idx = np.clip(indices - 1, 0, len(gt_times) - 1)
    dt_before = np.abs(gt_times[prev_idx] - lidar_times)
    dt_after  = np.abs(gt_times[indices]  - lidar_times)
    best = np.where(dt_before < dt_after, prev_idx, indices)
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


def compose_poses(T_list):
    poses = [np.eye(4)]
    for T in T_list:
        poses.append(poses[-1] @ T)
    return poses


# ── Method runners ────────────────────────────────────────────────────────────

def run_iv_gicp(frames, device="cuda", window_size=1):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(
        voxel_size=1.0,
        source_voxel_size=0.5,
        alpha=0.5,                   # intensity: key for tunnel degeneracy recovery
        max_correspondence_distance=3.0,
        initial_threshold=3.0,
        min_motion_th=0.1,
        huber_delta=1.0,
        max_iterations=30,
        adaptive_voxelization=True,  # C1: entropy-based for tunnels
        max_map_frames=20,
        window_size=window_size,
        device=device,
        use_distribution_propagation=False,
    )

    abs_poses = [np.eye(4)]
    times = []
    reg_times = []
    map_times = []
    print(f"\n[IV-GICP] {len(frames)} frames  device={device}")
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy())
        times.append(elapsed)
        reg_times.append(result.reg_ms)
        map_times.append(result.map_ms)
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:6.0f}ms", end="\r")

    _print_timing("IV-GICP", times[1:])
    reg_arr = np.array(reg_times[1:])
    map_arr = np.array(map_times[1:])
    print(f"    reg={reg_arr.mean():.1f}ms  map={map_arr.mean():.1f}ms")
    rel_poses = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i]
                 for i in range(1, len(abs_poses))]
    return rel_poses, {"total": times[1:], "reg": reg_times[1:], "map": map_times[1:]}


def run_gicp_baseline(frames, device="cuda", window_size=1):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(
        voxel_size=0.5,
        source_voxel_size=0.25,
        alpha=0.0,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        huber_delta=1.0,
        max_iterations=30,
        adaptive_voxelization=False,
        max_map_frames=40,
        window_size=window_size,
        device=device,
        use_distribution_propagation=False,
    )

    abs_poses = [np.eye(4)]
    times = []
    reg_times = []
    map_times = []
    print(f"\n[GICP-Baseline] {len(frames)} frames  device={device}")
    for i, (ts, pts) in enumerate(frames):
        pts_no_i = pts.copy(); pts_no_i[:, 3] = 0.0
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts_no_i[:, :3], pts_no_i[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy())
        times.append(elapsed)
        reg_times.append(result.reg_ms)
        map_times.append(result.map_ms)
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:6.0f}ms", end="\r")

    _print_timing("GICP-Baseline", times[1:])
    rel_poses = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i]
                 for i in range(1, len(abs_poses))]
    return rel_poses, {"total": times[1:], "reg": reg_times[1:], "map": map_times[1:]}


def run_kiss_icp(frames):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    config = KISSConfig()
    config.data.deskew = False
    config.data.max_range = 80.0
    config.data.min_range = 0.5
    config.mapping.voxel_size = 1.5   # fast vehicle: KISS-ICP auto-tunes but start coarse

    od = KissICP(config=config)
    abs_poses = [np.eye(4)]
    times = []
    print(f"\n[KISS-ICP] {len(frames)} frames")
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(od.last_pose.copy())
        times.append(elapsed)
        if i % 200 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:5.1f}ms", end="\r")

    _print_timing("KISS-ICP", times)
    return abs_poses, {"total": times}


def _print_timing(name, times):
    times = np.array(times)
    print(f"\n  [{name}] mean={times.mean():.1f}ms  "
          f"median={np.median(times):.1f}ms  "
          f"std={times.std():.1f}ms  "
          f"→ {1000/times.mean():.2f} Hz")


def compute_ate(gt_tum: Path, pred_tum: Path) -> float | None:
    """Compute ATE RMSE (m) using evo Python API."""
    try:
        from evo.tools import file_interface
        from evo.core import sync, metrics
        traj_ref = file_interface.read_tum_trajectory_file(str(gt_tum))
        traj_est = file_interface.read_tum_trajectory_file(str(pred_tum))
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
        traj_est.align(traj_ref, correct_scale=False)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((traj_ref, traj_est))
        return float(ape.get_statistic(metrics.StatisticsType.rmse))
    except Exception as e:
        print(f"  evo ATE failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="01", choices=["01", "02", "03"],
                        help="Sequence number (01/02/03)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kiss-only", action="store_true")
    parser.add_argument("--no-gicp-baseline", action="store_true")
    parser.add_argument("--window-size", type=int, default=1,
                        help="FORM window smoothing (1=disabled, e.g. 10)")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    seq_name = f"Urban_Tunnel{args.seq}"
    bag_path = GEODE_ROOT / "sensor_data" / "Urban_tunnel" / seq_name / f"{seq_name}.bag"
    gt_path  = GEODE_ROOT / "groundtruth" / "Urban_tunnel" / f"{seq_name}.txt"
    out = Path(__file__).parent.parent / "results" / "geode" / seq_name
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    frames = read_geode_frames(bag_path, max_frames=args.max_frames)
    n = len(frames)
    duration = frames[-1][0] - frames[0][0]
    timestamps = [f[0] for f in frames]

    # ── Load and save GT ──────────────────────────────────────────────────────
    print(f"\nLoading GT: {gt_path}")
    gt_times, gt_poses = load_geode_gt(gt_path)
    gt_at_lidar = interpolate_gt_at_lidar_times(gt_times, gt_poses, np.array(timestamps))
    save_tum(gt_at_lidar, timestamps, out / "gt.tum")
    gt_path_len, gt_end_disp = traj_stats(gt_at_lidar)
    print(f"  GT: path={gt_path_len:.1f}m  end_disp={gt_end_disp:.1f}m")

    results = {}

    # ── KISS-ICP ──────────────────────────────────────────────────────────────
    ki_poses, ki_timing = run_kiss_icp(frames)
    save_tum(ki_poses, timestamps, out / "kiss_icp.tum")
    path_len, end_disp = traj_stats(ki_poses)
    ate = compute_ate(out / "gt.tum", out / "kiss_icp.tum")
    results["KISS-ICP"] = {
        "mean_ms": float(np.mean(ki_timing["total"])),
        "hz": float(1000 / np.mean(ki_timing["total"])),
        "path_length_m": path_len, "end_displacement_m": end_disp,
        "ate_rmse_m": ate, "n_frames": n,
    }

    if not args.kiss_only:
        # ── IV-GICP ───────────────────────────────────────────────────────────
        iv_rel, iv_timing = run_iv_gicp(frames, device=device, window_size=args.window_size)
        iv_poses = compose_poses(iv_rel)
        save_tum(iv_poses, timestamps, out / "iv_gicp.tum")
        path_len, end_disp = traj_stats(iv_poses)
        ate = compute_ate(out / "gt.tum", out / "iv_gicp.tum")
        results["IV-GICP"] = {
            "mean_ms": float(np.mean(iv_timing["total"])),
            "hz": float(1000 / np.mean(iv_timing["total"])),
            "reg_ms": float(np.mean(iv_timing["reg"])),
            "map_ms": float(np.mean(iv_timing["map"])),
            "path_length_m": path_len, "end_displacement_m": end_disp,
            "ate_rmse_m": ate, "n_frames": n, "device": device,
        }

        # ── GICP Baseline ─────────────────────────────────────────────────────
        if not args.no_gicp_baseline:
            gb_rel, gb_timing = run_gicp_baseline(frames, device=device, window_size=args.window_size)
            gb_poses = compose_poses(gb_rel)
            save_tum(gb_poses, timestamps, out / "gicp_baseline.tum")
            path_len, end_disp = traj_stats(gb_poses)
            ate = compute_ate(out / "gt.tum", out / "gicp_baseline.tum")
            results["GICP-Baseline"] = {
                "mean_ms": float(np.mean(gb_timing["total"])),
                "hz": float(1000 / np.mean(gb_timing["total"])),
                "reg_ms": float(np.mean(gb_timing["reg"])),
                "map_ms": float(np.mean(gb_timing["map"])),
                "path_length_m": path_len, "end_displacement_m": end_disp,
                "ate_rmse_m": ate, "n_frames": n, "device": device,
            }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  GEODE {seq_name}  |  {n} frames  |  {duration:.1f}s  |  "
          f"avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame")
    print(f"  GT:  path={gt_path_len:.1f}m  end_disp={gt_end_disp:.2f}m")
    print(f"{'='*76}")
    print(f"  {'Method':<22} {'ms/f':>6} {'Hz':>6} {'reg ms':>7} {'map ms':>7} "
          f"{'Path(m)':>9} {'End(m)':>8} {'ATE RMSE':>10} {'Device':>7}")
    print(f"  {'-'*88}")
    for name, r in results.items():
        ate_str = f"{r['ate_rmse_m']:.3f}m" if r.get("ate_rmse_m") else "  N/A   "
        dev = r.get("device", "cpu")
        reg_s = f"{r['reg_ms']:.1f}" if "reg_ms" in r else "  N/A"
        map_s = f"{r['map_ms']:.1f}" if "map_ms" in r else "  N/A"
        print(f"  {name:<22} {r['mean_ms']:>6.1f} {r['hz']:>6.2f} {reg_s:>7} {map_s:>7} "
              f"{r['path_length_m']:>9.1f} {r['end_displacement_m']:>8.2f} "
              f"{ate_str:>10} {dev:>7}")
    print(f"{'='*76}")
    print(f"\n  Trajectories → {out}")
    print(f"  ATE eval:  evo_ape tum {out}/gt.tum {out}/iv_gicp.tum --align --plot\n")

    meta = {"seq": seq_name, "bag": str(bag_path), "n_frames": n,
            "duration_s": duration,
            "avg_pts_per_frame": float(np.mean([len(f[1]) for f in frames])),
            "gt_path_m": gt_path_len, "gt_end_disp_m": gt_end_disp}
    with open(out / "results.json", "w") as f:
        json.dump({"meta": meta, "methods": results}, f, indent=2)
    print(f"  Results JSON → {out}/results.json\n")
    return results


if __name__ == "__main__":
    main()
