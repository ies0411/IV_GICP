"""
Hilti 2022 Evaluation Script
=============================
Evaluates IV-GICP vs KISS-ICP on Hilti SLAM Challenge 2022
sequence: exp07_long_corridor (long corridor, geometric degeneracy scenario)

Usage:
    uv run python examples/run_hilti_eval.py [--bag PATH] [--max-frames N] [--device auto]
    uv run python examples/run_hilti_eval.py --kiss-only           # KISS-ICP full run
    uv run python examples/run_hilti_eval.py --max-frames 600      # IV-GICP subset

Output:
    - results/hilti/*.tum  (TUM format, evaluate with: evo_ape tum gt.tum pred.tum --align)
    - results/hilti/timing_results.json
    - results/hilti/component_timing.json  (breakdown: load / voxelize / kdtree / gn)

Note: ATE/RPE requires Hilti GT from hilti-challenge.com
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

BAG_PATH = "/home/km/data/hilti/2022/exp07_long_corridor.bag"
LIDAR_TOPIC = "/hesai/pandar"

# ── Hilti bag reader ──────────────────────────────────────────────────────────

def read_hilti_frames(bag_path: str = BAG_PATH, max_frames: int = None):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    frames = []
    print(f"Reading: {bag_path}")
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = sum(1 for _ in bag.messages(connections=conns))
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        print(f"  LiDAR frames in bag: {total}")
        for conn, ts_ns, rawdata in bag.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            pts = _parse_pc2(msg)
            if pts is not None and len(pts) > 200:
                frames.append((ts_ns / 1e9, pts))
                if len(frames) % 200 == 0:
                    print(f"  Loaded {len(frames)}/{total} ...", end="\r")
                if max_frames and len(frames) >= max_frames:
                    break
    print(f"\n  Done: {len(frames)} frames  "
          f"({frames[-1][0]-frames[0][0]:.1f}s, "
          f"avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame)")
    return frames


def _parse_pc2(msg) -> np.ndarray:
    """PointCloud2 → (N,4) float64 [x,y,z,intensity]. Hesai Pandar field offsets."""
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    x   = data[:, 0:4].view(np.float32).reshape(-1)
    y   = data[:, 4:8].view(np.float32).reshape(-1)
    z   = data[:, 8:12].view(np.float32).reshape(-1)
    itn = data[:, 16:20].view(np.float32).reshape(-1)
    pts = np.stack([x, y, z, itn], axis=1).astype(np.float64)
    valid = (np.isfinite(pts).all(axis=1)
             & (np.linalg.norm(pts[:, :3], axis=1) > 0.5)
             & (np.linalg.norm(pts[:, :3], axis=1) < 80.0))
    return pts[valid]


# ── Pose / trajectory utilities ───────────────────────────────────────────────

def compose_poses(T_list):
    poses = [np.eye(4)]
    for T in T_list:
        poses.append(poses[-1] @ T)
    return poses


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
    """Total path length and end displacement."""
    dists = [np.linalg.norm(poses[i+1][:3,3] - poses[i][:3,3])
             for i in range(len(poses)-1)]
    path_len = float(np.sum(dists))
    end_disp  = float(np.linalg.norm(poses[-1][:3,3] - poses[0][:3,3]))
    return path_len, end_disp


# ── IV-GICP runner ────────────────────────────────────────────────────────────

def run_iv_gicp(frames, device="cuda"):
    """
    IV-GICP via IVGICPPipeline (growing map, constant-velocity init, fast FlatVoxelMap).

    Uses IVGICPPipeline instead of IVGICP.register() frame-to-frame:
      - Accumulates a growing voxel map (better correspondences)
      - Constant-velocity motion prediction for initial pose
      - FlatVoxelMap: O(N_frame) incremental updates vs O(N_acc × depth) octree rebuild
      - Source voxel downsampling (0.5m) reduces GN input from 47k → ~5k points
    """
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(
        voxel_size=0.3,               # fine voxels: preserve corridor features (doors, pillars)
        source_voxel_size=0.2,        # centroid downsample: 51k → ~2-5k pts, keeps structure
        alpha=0.5,                    # higher alpha: intensity critical in degenerate corridor
        max_correspondence_distance=0.5,  # tight: robot moves 0.15m/frame, avoid false matches
        initial_threshold=0.15,       # 3σ=0.45m sanity bound, matches per-frame motion
        min_motion_th=0.03,
        max_iterations=20,
        device=device,
        use_distribution_propagation=False,
    )

    abs_poses = [np.eye(4)]
    times_total = []

    print(f"\n[IV-GICP Pipeline] {len(frames)} frames  device={device}")
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        t_total = (time.perf_counter() - t0) * 1000

        abs_poses.append(result.pose.copy())
        times_total.append(t_total)

        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {t_total:6.0f}ms", end="\r")

    _print_timing("IV-GICP", times_total[1:])  # skip first frame (map init only)
    # Convert absolute poses to relative for compose_poses compatibility
    rel_poses = []
    for i in range(1, len(abs_poses)):
        rel_poses.append(np.linalg.inv(abs_poses[i-1]) @ abs_poses[i])
    return rel_poses, {"total": times_total[1:]}


# ── KISS-ICP runner ───────────────────────────────────────────────────────────

def run_kiss_icp(frames):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    config = KISSConfig()
    config.data.deskew = False
    config.data.max_range = 80.0
    config.data.min_range = 0.5
    config.mapping.voxel_size = 0.5

    od = KissICP(config=config)
    abs_poses = [np.eye(4)]
    times = []

    print(f"\n[KISS-ICP] {len(frames)} frames")
    for i, (ts, pts) in enumerate(frames):
        src = pts[:, :3].astype(np.float64)
        t0 = time.perf_counter()
        od.register_frame(src, np.full(len(src), ts))
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(od.last_pose.copy())
        times.append(elapsed)
        if i % 200 == 0 or i == len(frames)-1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:5.1f}ms", end="\r")

    _print_timing("KISS-ICP", times)
    return abs_poses, {"total": times}


# ── GICP Baseline runner (GPU, alpha=0) ───────────────────────────────────────

def run_gicp_baseline(frames, device="cuda"):
    """GICP baseline via pipeline with alpha=0 (geometry only, no intensity)."""
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from iv_gicp.pipeline import IVGICPPipeline

    pipeline = IVGICPPipeline(
        voxel_size=1.0,
        source_voxel_size=0.5,
        alpha=0.0,
        max_iterations=20,
        device=device,
        use_distribution_propagation=False,
    )

    abs_poses = [np.eye(4)]
    times = []
    print(f"\n[GICP-Baseline Pipeline] {len(frames)} frames  device={device}")
    for i, (ts, pts) in enumerate(frames):
        pts_no_i = pts.copy(); pts_no_i[:, 3] = 0.0
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts_no_i[:, :3], pts_no_i[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy())
        times.append(elapsed)
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {elapsed:6.0f}ms", end="\r")

    _print_timing("GICP-Baseline", times[1:])
    rel_poses = []
    for i in range(1, len(abs_poses)):
        rel_poses.append(np.linalg.inv(abs_poses[i-1]) @ abs_poses[i])
    return rel_poses, {"total": times[1:]}


def _print_timing(name, times):
    times = np.array(times)
    print(f"\n  [{name}] mean={times.mean():.1f}ms  "
          f"median={np.median(times):.1f}ms  "
          f"std={times.std():.1f}ms  "
          f"min={times.min():.1f}ms  max={times.max():.1f}ms  "
          f"→ {1000/times.mean():.2f} Hz")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default=BAG_PATH)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kiss-only", action="store_true",
                        help="Run KISS-ICP only (fast, full sequence)")
    parser.add_argument("--no-gicp-baseline", action="store_true",
                        help="Skip slow GICP baseline")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    out = Path(__file__).parent.parent / "results" / "hilti"
    out.mkdir(parents=True, exist_ok=True)

    frames = read_hilti_frames(args.bag, max_frames=args.max_frames)
    n = len(frames)
    duration = frames[-1][0] - frames[0][0]
    timestamps = [f[0] for f in frames]
    results = {}

    # ── KISS-ICP (fast, run always / full sequence) ───────────────────────────
    ki_poses, ki_timing = run_kiss_icp(frames)
    save_tum(ki_poses, timestamps, out / "kiss_icp.tum")
    path_len, end_disp = traj_stats(ki_poses)
    results["KISS-ICP"] = {
        "mean_ms":  float(np.mean(ki_timing["total"])),
        "median_ms": float(np.median(ki_timing["total"])),
        "std_ms":   float(np.std(ki_timing["total"])),
        "hz":       float(1000 / np.mean(ki_timing["total"])),
        "path_length_m": path_len,
        "end_displacement_m": end_disp,
        "n_frames": n,
    }

    if not args.kiss_only:
        # ── IV-GICP ──────────────────────────────────────────────────────────
        iv_rel, iv_timing = run_iv_gicp(frames, device=device)
        iv_poses = compose_poses(iv_rel)
        save_tum(iv_poses, timestamps, out / "iv_gicp.tum")
        path_len, end_disp = traj_stats(iv_poses)
        results["IV-GICP"] = {
            "mean_ms":  float(np.mean(iv_timing["total"])),
            "median_ms": float(np.median(iv_timing["total"])),
            "std_ms":   float(np.std(iv_timing["total"])),
            "hz":       float(1000 / np.mean(iv_timing["total"])),
            "path_length_m": path_len,
            "end_displacement_m": end_disp,
            "n_frames": n,
            "device": device,
        }

        # ── GICP Baseline ─────────────────────────────────────────────────────
        if not args.no_gicp_baseline:
            gb_rel, gb_timing = run_gicp_baseline(frames, device=device)
            gb_poses = compose_poses(gb_rel)
            save_tum(gb_poses, timestamps, out / "gicp_baseline.tum")
            path_len, end_disp = traj_stats(gb_poses)
            results["GICP-Baseline"] = {
                "mean_ms":  float(np.mean(gb_timing["total"])),
                "median_ms": float(np.median(gb_timing["total"])),
                "std_ms":   float(np.std(gb_timing["total"])),
                "hz":       float(1000 / np.mean(gb_timing["total"])),
                "path_length_m": path_len,
                "end_displacement_m": end_disp,
                "n_frames": n,
                "device": device,
            }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Hilti exp07_long_corridor  |  {n} frames  |  {duration:.1f}s  |  "
          f"avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame")
    print(f"{'='*72}")
    print(f"  {'Method':<22} {'Mean(ms)':>9} {'Hz':>7} {'Path(m)':>9} {'EndDisp(m)':>11} {'Device':>7}")
    print(f"  {'-'*70}")
    for name, r in results.items():
        dev = r.get("device", "cpu")
        print(f"  {name:<22} {r['mean_ms']:>9.1f} {r['hz']:>7.2f} "
              f"{r['path_length_m']:>9.1f} {r['end_displacement_m']:>11.2f} {dev:>7}")
    print(f"{'='*72}")
    print(f"\n  Trajectories: {out}")
    print(f"  Eval:  evo_ape tum <gt.tum> {out}/iv_gicp.tum --align --plot")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    meta = {"bag": args.bag, "n_frames": n, "duration_s": duration,
            "avg_pts_per_frame": float(np.mean([len(f[1]) for f in frames]))}
    with open(out / "results.json", "w") as f:
        json.dump({"meta": meta, "methods": results}, f, indent=2)
    print(f"  Results: {out}/results.json\n")
    return results


if __name__ == "__main__":
    main()
