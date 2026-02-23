"""
Hilti 2022 Evaluation Script
=============================
Evaluates IV-GICP vs GICP Baseline vs KISS-ICP on Hilti SLAM Challenge 2022
sequence: exp07_long_corridor (long corridor, geometric degeneracy scenario)

Usage:
    uv run python examples/run_hilti_eval.py [--bag PATH] [--max-frames N] [--device auto]

Output:
    - results/hilti_trajectories/*.tum  (TUM format for evo evaluation)
    - results/hilti_timing.json
    - Prints summary table

Note: ATE/RPE requires Hilti GT poses (download from hilti-challenge.com).
      This script saves trajectories in TUM format for later evaluation with evo.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

# ── Hilti bag reader ─────────────────────────────────────────────────────────

def read_hilti_frames(bag_path: str, topic: str = "/hesai/pandar", max_frames: int = None):
    """
    Read LiDAR frames from Hilti rosbag.
    Returns list of (timestamp_sec, points_Nx4) where points[:,3] = intensity.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    frames = []

    print(f"Reading bag: {bag_path}")
    with Reader(bag_path) as bag:
        lidar_conns = [c for c in bag.connections if c.topic == topic]
        total = sum(1 for _ in bag.messages(connections=lidar_conns))
        print(f"  Total LiDAR frames: {total}")

        lidar_conns = [c for c in bag.connections if c.topic == topic]
        count = 0
        for conn, ts_ns, rawdata in bag.messages(connections=lidar_conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            pts = _parse_pointcloud2(msg)
            if pts is not None and len(pts) > 100:
                frames.append((ts_ns / 1e9, pts))
                count += 1
                if count % 100 == 0:
                    print(f"  Loaded {count}/{total} frames ...", end="\r")
                if max_frames and count >= max_frames:
                    break

    print(f"\n  Done. Loaded {len(frames)} frames.")
    return frames


def _parse_pointcloud2(msg) -> np.ndarray:
    """Parse PointCloud2 msg → (N,4) float64 array [x,y,z,intensity]."""
    # Hesai Pandar fields: x@0(f32), y@4(f32), z@8(f32), intensity@16(f32)
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n_pts = msg.width * msg.height
    step = msg.point_step
    data = data.reshape(n_pts, step)

    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    intensity = data[:, 16:20].view(np.float32).reshape(-1)

    pts = np.stack([x, y, z, intensity], axis=1).astype(np.float64)

    # Filter NaN / inf / near-zero points
    valid = (
        np.isfinite(pts).all(axis=1)
        & (np.linalg.norm(pts[:, :3], axis=1) > 0.5)
        & (np.linalg.norm(pts[:, :3], axis=1) < 100.0)
    )
    return pts[valid]


# ── Pose utilities ────────────────────────────────────────────────────────────

def compose_poses(T_list):
    """Compose list of relative transforms → list of global poses."""
    poses = [np.eye(4)]
    for T in T_list:
        poses.append(poses[-1] @ T)
    return poses


def save_tum(poses, timestamps, path: Path):
    """Save poses in TUM format: timestamp tx ty tz qx qy qz qw"""
    from scipy.spatial.transform import Rotation
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"  Saved: {path}")


def compute_rpe(poses, step=1):
    """
    Compute RPE (relative pose error) without ground truth.
    Here we compute frame-to-frame translation magnitude as a drift proxy.
    Returns mean/std of per-step translation magnitudes.
    """
    trans = []
    for i in range(0, len(poses) - step, step):
        dT = np.linalg.inv(poses[i]) @ poses[i + step]
        trans.append(np.linalg.norm(dT[:3, 3]))
    return np.array(trans)


# ── Method runners ────────────────────────────────────────────────────────────

def run_iv_gicp(frames, device="cuda"):
    """Run IV-GICP on frame sequence. Returns (relative_poses, times_ms)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp import IVGICP

    gicp = IVGICP(device=device)
    rel_poses = []
    times_ms = []

    print(f"\n[IV-GICP] Running on {len(frames)-1} frame pairs (device={device})...")
    for i in range(1, len(frames)):
        src_pts = frames[i][1]
        tgt_pts = frames[i-1][1]
        t0 = time.perf_counter()
        T = gicp.register(src_pts, tgt_pts)
        elapsed = (time.perf_counter() - t0) * 1000
        rel_poses.append(T)
        times_ms.append(elapsed)
        if i % 50 == 0:
            print(f"  Frame {i}/{len(frames)-1}  {elapsed:.0f}ms", end="\r")

    print(f"\n  Done. Mean: {np.mean(times_ms):.1f}ms  Median: {np.median(times_ms):.1f}ms")
    return rel_poses, times_ms


def run_gicp_baseline(frames, device="cpu"):
    """Run GICP baseline (no adaptive voxelization, no intensity) on frame sequence."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp import IVGICP

    # Baseline: alpha=0 (no intensity), fixed voxel size
    gicp = IVGICP(device=device, alpha=0.0)
    rel_poses = []
    times_ms = []

    print(f"\n[GICP Baseline] Running on {len(frames)-1} frame pairs...")
    for i in range(1, len(frames)):
        src_pts = frames[i][1][:, :3]  # geometry only
        tgt_pts = frames[i-1][1][:, :3]
        # Pad with dummy intensity
        src_pts4 = np.hstack([src_pts, np.zeros((len(src_pts), 1))])
        tgt_pts4 = np.hstack([tgt_pts, np.zeros((len(tgt_pts), 1))])
        t0 = time.perf_counter()
        T = gicp.register(src_pts4, tgt_pts4)
        elapsed = (time.perf_counter() - t0) * 1000
        rel_poses.append(T)
        times_ms.append(elapsed)
        if i % 50 == 0:
            print(f"  Frame {i}/{len(frames)-1}  {elapsed:.0f}ms", end="\r")

    print(f"\n  Done. Mean: {np.mean(times_ms):.1f}ms  Median: {np.median(times_ms):.1f}ms")
    return rel_poses, times_ms


def run_kiss_icp(frames):
    """Run KISS-ICP (low-level KissICP class) on frame sequence."""
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    config = KISSConfig()
    config.data.deskew = False
    config.data.max_range = 60.0
    config.data.min_range = 0.5
    # voxel_size must be set; auto-estimate from point density not available without dataset wrapper
    config.mapping.voxel_size = 0.5

    odometry = KissICP(config=config)
    rel_poses = []
    abs_poses = [np.eye(4)]
    times_ms = []

    print(f"\n[KISS-ICP] Running on {len(frames)} frames...")
    for i, (ts, pts) in enumerate(frames):
        src = pts[:, :3].astype(np.float64)
        t0 = time.perf_counter()
        odometry.register_frame(src, np.full(len(src), ts))
        elapsed = (time.perf_counter() - t0) * 1000

        cur_pose = odometry.last_pose.copy()
        if i > 0:
            rel = np.linalg.inv(abs_poses[-1]) @ cur_pose
            rel_poses.append(rel)
            times_ms.append(elapsed)
        abs_poses.append(cur_pose)

        if i % 50 == 0:
            print(f"  Frame {i}/{len(frames)}  {elapsed:.0f}ms", end="\r")

    print(f"\n  Done. Mean: {np.mean(times_ms):.1f}ms  Median: {np.median(times_ms):.1f}ms")
    return rel_poses, times_ms, abs_poses


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="/home/km/data/hilti/2022/exp07_long_corridor.bag")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (None = all 1322)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip", type=int, default=1,
                        help="Use every Nth frame (default=1, all frames)")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    out_dir = Path(__file__).parent.parent / "results" / "hilti"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frames
    frames = read_hilti_frames(args.bag, max_frames=args.max_frames)
    if args.skip > 1:
        frames = frames[::args.skip]
        print(f"  Subsampled to {len(frames)} frames (every {args.skip}th)")

    timestamps = [f[0] for f in frames]
    n_frames = len(frames)
    print(f"\nTotal frames for evaluation: {n_frames}")

    results = {}

    # 2. Run IV-GICP
    iv_rel, iv_times = run_iv_gicp(frames, device=device)
    iv_poses = compose_poses(iv_rel)
    save_tum(iv_poses, timestamps, out_dir / "iv_gicp.tum")
    results["IV-GICP"] = {
        "mean_ms": float(np.mean(iv_times)),
        "median_ms": float(np.median(iv_times)),
        "std_ms": float(np.std(iv_times)),
        "n_frames": len(iv_times),
    }

    # 3. Run GICP Baseline
    gb_rel, gb_times = run_gicp_baseline(frames, device="cpu")
    gb_poses = compose_poses(gb_rel)
    save_tum(gb_poses, timestamps, out_dir / "gicp_baseline.tum")
    results["GICP-Baseline"] = {
        "mean_ms": float(np.mean(gb_times)),
        "median_ms": float(np.median(gb_times)),
        "std_ms": float(np.std(gb_times)),
        "n_frames": len(gb_times),
    }

    # 4. Run KISS-ICP
    ki_rel, ki_times, ki_abs_poses = run_kiss_icp(frames)
    save_tum(ki_abs_poses, timestamps, out_dir / "kiss_icp.tum")
    results["KISS-ICP"] = {
        "mean_ms": float(np.mean(ki_times)),
        "median_ms": float(np.median(ki_times)),
        "std_ms": float(np.std(ki_times)),
        "n_frames": len(ki_times),
    }

    # 5. Trajectory drift proxy: end-to-end displacement
    def end_disp(poses):
        return float(np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]))

    for name, poses in [("IV-GICP", iv_poses), ("GICP-Baseline", gb_poses),
                         ("KISS-ICP", ki_abs_poses)]:
        results[name]["end_displacement_m"] = end_disp(poses)

    # 6. Print summary
    print("\n" + "="*65)
    print(f"  Hilti exp07_long_corridor  ({n_frames} frames, {(timestamps[-1]-timestamps[0]):.1f}s)")
    print("="*65)
    print(f"  {'Method':<20} {'Mean ms':>9} {'Median ms':>10} {'End-disp (m)':>14}")
    print("-"*65)
    for name, r in results.items():
        print(f"  {name:<20} {r['mean_ms']:>9.1f} {r['median_ms']:>10.1f} {r['end_displacement_m']:>14.2f}")
    print("="*65)
    print("\n  Note: ATE/RPE requires Hilti GT poses.")
    print(f"  Trajectories saved to: {out_dir}")
    print("  Evaluate with: evo_ape tum <gt.tum> <pred.tum> --align")

    # 7. Save JSON
    json_path = out_dir / "timing_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Timing saved: {json_path}\n")

    return results


if __name__ == "__main__":
    main()
