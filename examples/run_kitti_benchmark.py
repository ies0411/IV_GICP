#!/usr/bin/env python3
"""
KITTI Odometry Benchmark: IV-GICP vs KISS-ICP vs GenZ-ICP vs GICP-Baseline

Usage:
    uv run python examples/run_kitti_benchmark.py --seq 08 [--max-frames 300]
    uv run python examples/run_kitti_benchmark.py --seq 00 --max-frames 500

Data root assumed: /home/km/data/kitti/dataset
  sequences/XX/velodyne/*.bin   — LiDAR scans
  poses/XX.txt                  — GT poses (camera frame, 3x4 row-major)

Output:
    results/kitti/seqXX/  — per-method TUM files + results.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

KITTI_ROOT   = Path("/home/km/data/kitti/dataset")
GENZ_BINARY  = Path(__file__).parent.parent / "thirdparty/genz-icp/kitti_runner/build/genz_kitti_runner"
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "kitti"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_kitti_sequence(seq: str, max_frames: int = None):
    """Load velodyne .bin files and GT poses. Returns (frames_xyzI, poses_gt)."""
    velo_dir = KITTI_ROOT / "sequences" / seq / "velodyne"
    pose_file = KITTI_ROOT / "poses" / f"{seq}.txt"

    bins = sorted(velo_dir.glob("*.bin"))
    if max_frames:
        bins = bins[:max_frames]

    frames = []
    for bf in bins:
        data = np.fromfile(str(bf), dtype=np.float32).reshape(-1, 4)
        r = np.linalg.norm(data[:, :3], axis=1)
        mask = (r > 0.5) & (r < 80.0)
        frames.append(data[mask].astype(np.float64))

    poses_gt = None
    if pose_file.exists():
        poses_gt = []
        with open(pose_file) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 12:
                    continue
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses_gt.append(T)
        if max_frames:
            poses_gt = poses_gt[:max_frames]

    return frames, poses_gt


# ── Metrics ───────────────────────────────────────────────────────────────────

def ate_rmse(poses_est, poses_gt):
    """ATE RMSE after Umeyama alignment (handles camera/LiDAR frame difference)."""
    n = min(len(poses_est), len(poses_gt))
    if n < 2:
        return float("nan"), float("nan")
    t_est = np.array([p[:3, 3] for p in poses_est[:n]])
    t_gt  = np.array([p[:3, 3] for p in poses_gt[:n]])

    mu_e = t_est.mean(0); mu_g = t_gt.mean(0)
    H = (t_est - mu_e).T @ (t_gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ D @ U.T
    t = mu_g - R @ mu_e

    errs = np.array([np.linalg.norm(R @ p[:3,3] + t - gt[:3,3])
                     for p, gt in zip(poses_est[:n], poses_gt[:n])])
    return float(np.sqrt(np.mean(errs**2))), float(np.mean(errs))


def kitti_t_err(poses_est, poses_gt, step=10, lengths=(100,200,300,400,500,600,700,800)):
    """KITTI standard translation error (%) over fixed-distance segments."""
    n = min(len(poses_est), len(poses_gt))
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i-1] + np.linalg.norm(poses_gt[i][:3,3] - poses_gt[i-1][:3,3])

    t_errs = []
    for i in range(0, n, step):
        for L in lengths:
            j = np.searchsorted(cum, cum[i] + L)
            if j >= n: continue
            actual = cum[j] - cum[i]
            if actual < 1.0: continue
            T_est = np.linalg.inv(poses_est[i]) @ poses_est[j]
            T_gt  = np.linalg.inv(poses_gt[i])  @ poses_gt[j]
            T_err = np.linalg.inv(T_gt) @ T_est
            t_errs.append(np.linalg.norm(T_err[:3,3]) / actual * 100.0)

    return float(np.mean(t_errs)) if t_errs else float("nan")


def print_timing(name, times_ms):
    t = np.array(times_ms)
    print(f"  [{name}] mean={t.mean():.1f}ms  median={np.median(t):.1f}ms  "
          f"std={t.std():.1f}ms  → {1000/t.mean():.2f} Hz")


# ── Method runners ────────────────────────────────────────────────────────────

def run_iv_gicp(frames, device, alpha, label, **kw):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=1.0,               # standard outdoor voxel size
        source_voxel_size=0.3,        # KITTI ~120k pts → ~5-15k after 0.3m downsample
        min_points_per_voxel=5,       # require 5+ pts for reliable covariance (rank 3)
        alpha=alpha,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        min_motion_th=0.05,
        max_iterations=30,            # 30 GN iterations; typical convergence <20
        max_map_points=100_000,
        max_map_frames=10,            # 10-frame window: ~170m coverage at KITTI speed
        adaptive_voxelization=False,  # C1 disabled for outdoor (geometry-rich, large map)
        device=device,
        use_distribution_propagation=False,
        **kw,
    )
    abs_poses = []
    times = []
    reg_times = []
    map_times = []
    print(f"\n[{label}] {len(frames)} frames  device={device}  α={alpha}")
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        times.append((time.perf_counter() - t0) * 1000)
        reg_times.append(result.reg_ms)
        map_times.append(result.map_ms)
        abs_poses.append(result.pose.copy())
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {times[-1]:6.1f}ms", end="\r")
    print()
    print_timing(label, times[1:])
    reg_arr = np.array(reg_times[1:])
    map_arr = np.array(map_times[1:])
    print(f"    reg={reg_arr.mean():.1f}ms  map={map_arr.mean():.1f}ms  "
          f"other={times[1:][-1] - reg_times[-1] - map_times[-1]:.1f}ms (last frame)")
    return abs_poses, times, reg_times, map_times


def run_kiss_icp(frames):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.max_range = 80.0
    cfg.data.min_range = 0.5
    cfg.data.deskew    = False
    cfg.mapping.voxel_size = 1.0
    od = KissICP(config=cfg)

    abs_poses = []
    times = []
    print(f"\n[KISS-ICP] {len(frames)} frames")
    for i, f in enumerate(frames):
        src = f[:, :3].astype(np.float64)
        t0 = time.perf_counter()
        od.register_frame(src, np.full(len(src), float(i)))
        times.append((time.perf_counter() - t0) * 1000)
        abs_poses.append(od.last_pose.copy())
        if i % 100 == 0 or i == len(frames) - 1:
            print(f"  {i:4d}/{len(frames)}  {times[-1]:5.1f}ms", end="\r")
    print()
    print_timing("KISS-ICP", times)
    return abs_poses, times


def run_genz_icp(seq: str, max_frames: int, out_dir: Path):
    """Run GenZ-ICP C++ binary and load the resulting poses."""
    if not GENZ_BINARY.exists():
        print(f"  [GenZ-ICP] binary not found: {GENZ_BINARY}")
        print("  Build with: cd thirdparty/genz-icp/kitti_runner/build && make -j")
        return None, []

    velo_dir = KITTI_ROOT / "sequences" / seq / "velodyne"
    poses_file = out_dir / "genz_icp.txt"
    cmd = [str(GENZ_BINARY), str(velo_dir), str(poses_file)]
    if max_frames:
        cmd.append(str(max_frames))

    print(f"\n[GenZ-ICP] running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  [GenZ-ICP] FAILED (returncode={result.returncode})")
        return None, []

    # Load poses
    poses = []
    with open(poses_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)

    n = len(poses)
    mean_ms = elapsed / n * 1000 if n > 0 else 0
    print(f"  [{n} poses loaded]  mean={mean_ms:.1f}ms  → {1000/mean_ms:.2f} Hz")
    return poses, []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",         default="08",  help="KITTI sequence ID")
    parser.add_argument("--max-frames",  type=int, default=None)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--skip-genz",   action="store_true", help="Skip GenZ-ICP (slow build)")
    parser.add_argument("--window-size", type=int, default=1,
                        help="FORM window smoothing size (1=disabled, 5-15 for real-time)")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    out = RESULTS_ROOT / f"seq{args.seq}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  KITTI Odometry Benchmark — seq {args.seq}")
    print(f"  data: {KITTI_ROOT}")
    print(f"  device: {device}  max_frames: {args.max_frames}  window_size: {args.window_size}")
    print(f"{'='*70}")

    frames, poses_gt = load_kitti_sequence(args.seq, args.max_frames)
    n = len(frames)
    if poses_gt:
        poses_gt = poses_gt[:n]
    print(f"\n  {n} frames loaded  GT: {'yes' if poses_gt else 'no'}")

    results = {}

    # ── KISS-ICP ─────────────────────────────────────────────────────────────
    ki_poses, ki_times = run_kiss_icp(frames)
    ki_ate, ki_ate_m = ate_rmse(ki_poses, poses_gt) if poses_gt else (float("nan"), float("nan"))
    ki_kt = kitti_t_err(ki_poses, poses_gt) if poses_gt else float("nan")
    results["KISS-ICP"] = dict(
        ate_rmse=ki_ate, ate_mean=ki_ate_m, kitti_t_err=ki_kt,
        mean_ms=float(np.mean(ki_times)), n_frames=n, device="cpu",
    )

    # ── GICP Baseline (alpha=0) ───────────────────────────────────────────────
    gi_poses, gi_times, gi_reg, gi_map = run_iv_gicp(frames, device, alpha=0.0, label="GICP-Baseline",
                                                      window_size=args.window_size)
    gi_ate, gi_ate_m = ate_rmse(gi_poses, poses_gt) if poses_gt else (float("nan"), float("nan"))
    gi_kt = kitti_t_err(gi_poses, poses_gt) if poses_gt else float("nan")
    results["GICP-Baseline"] = dict(
        ate_rmse=gi_ate, ate_mean=gi_ate_m, kitti_t_err=gi_kt,
        mean_ms=float(np.mean(gi_times[1:])), n_frames=n, device=device,
        reg_ms=float(np.mean(gi_reg[1:])), map_ms=float(np.mean(gi_map[1:])),
    )

    # ── IV-GICP (alpha=0.1) ───────────────────────────────────────────────────
    # α=0.5 → omega_I≈225 >> Omega_geo_xx≈1 → intensity dominates geometry → bad.
    # α=0.1 → omega_I≈9 ≈ Omega_geo_xx → balanced geo+intensity contribution.
    iv_poses, iv_times, iv_reg, iv_map = run_iv_gicp(frames, device, alpha=0.1, label="IV-GICP",
                                                      window_size=args.window_size)
    iv_ate, iv_ate_m = ate_rmse(iv_poses, poses_gt) if poses_gt else (float("nan"), float("nan"))
    iv_kt = kitti_t_err(iv_poses, poses_gt) if poses_gt else float("nan")
    results["IV-GICP"] = dict(
        ate_rmse=iv_ate, ate_mean=iv_ate_m, kitti_t_err=iv_kt,
        mean_ms=float(np.mean(iv_times[1:])), n_frames=n, device=device,
        reg_ms=float(np.mean(iv_reg[1:])), map_ms=float(np.mean(iv_map[1:])),
    )

    # ── GenZ-ICP ─────────────────────────────────────────────────────────────
    if not args.skip_genz:
        gz_poses, _ = run_genz_icp(args.seq, args.max_frames, out)
        if gz_poses:
            gz_ate, gz_ate_m = ate_rmse(gz_poses, poses_gt) if poses_gt else (float("nan"), float("nan"))
            gz_kt = kitti_t_err(gz_poses, poses_gt) if poses_gt else float("nan")
            results["GenZ-ICP"] = dict(
                ate_rmse=gz_ate, ate_mean=gz_ate_m, kitti_t_err=gz_kt,
                mean_ms=float("nan"), n_frames=len(gz_poses), device="cpu",
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  KITTI seq {args.seq}  |  {n} frames  |  GT: {'yes' if poses_gt else 'no'}")
    print(f"{'='*90}")
    print(f"  {'Method':<20} {'ATE(m)':>8} {'KITTI-t%':>9} {'total ms':>9} "
          f"{'reg ms':>8} {'map ms':>8} {'Hz':>6} {'Device':>7}")
    print(f"  {'-'*83}")
    for name, r in results.items():
        hz = 1000 / r['mean_ms'] if r['mean_ms'] > 0 else float("nan")
        reg = r.get('reg_ms', float("nan"))
        mmap = r.get('map_ms', float("nan"))
        print(f"  {name:<20} {r['ate_rmse']:>8.3f} {r['kitti_t_err']:>9.2f} "
              f"{r['mean_ms']:>9.1f} {reg:>8.1f} {mmap:>8.1f} {hz:>6.1f} "
              f"{r.get('device','?'):>7}")
    print(f"{'='*90}")

    with open(out / "results.json", "w") as f:
        json.dump({"seq": args.seq, "n_frames": n, "methods": results}, f, indent=2)
    print(f"\n  Saved: {out}/results.json")


if __name__ == "__main__":
    main()
