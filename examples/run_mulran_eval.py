#!/usr/bin/env python3
"""
MulRan Odometry Benchmark: IV-GICP vs KISS-ICP vs GICP-Baseline

Sequences available:
  /home/km/data/MulRan/DCC01/   — downtown campus (Ouster OS1-64)
  /home/km/data/MulRan/KAIST01/ — KAIST campus outdoor (Ouster) [needs extraction]

Point format: 4 float32/point (x, y, z, intensity)
  Many zero-padded points — filter by r > 0.5

GT format (global_pose.csv):
  timestamp, r11,r12,r13,tx, r21,r22,r23,ty, r31,r32,r33,tz   (rotation matrix)

Ouster timestamps from data_stamp.csv:
  timestamp,sensor_name  → filter sensor_name == "ouster"

Usage:
    uv run python examples/run_mulran_eval.py --seq DCC01 --max-frames 300
    uv run python examples/run_mulran_eval.py --seq DCC01 --max-frames 500 --alpha 0.1
    uv run python examples/run_mulran_eval.py --seq DCC01 --max-frames 300 --window-size 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

MULRAN_ROOT  = Path("/home/km/data/MulRan")
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "mulran"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mulran_frame(path: Path) -> np.ndarray:
    """Load one Ouster .bin file → (N,4) float64 xyzI, filtered."""
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = raw[:, :3].astype(np.float64)
    intensity = raw[:, 3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    mask = (r > 0.5) & (r < 100.0) & np.isfinite(r)
    pts = np.empty((mask.sum(), 4), dtype=np.float64)
    pts[:, :3] = xyz[mask]
    pts[:, 3]  = intensity[mask]
    return pts


def load_mulran_gt(gt_path: Path):
    """Load global_pose.csv → (timestamps_ns, poses list of 4x4)."""
    timestamps = []
    poses = []
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = list(map(float, line.split(',')))
        ts_ns = int(vals[0])
        # rotation matrix + translation: r11,r12,r13,tx, r21,r22,r23,ty, r31,r32,r33,tz
        R = np.array([
            [vals[1], vals[2],  vals[3]],
            [vals[5], vals[6],  vals[7]],
            [vals[9], vals[10], vals[11]],
        ])
        t = np.array([vals[4], vals[8], vals[12]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t
        timestamps.append(ts_ns)
        poses.append(T)
    return np.array(timestamps, dtype=np.int64), poses


def load_ouster_timestamps(seq_dir: Path):
    """Load data_stamp.csv → array of ouster scan timestamps (ns)."""
    ts_list = []
    for line in (seq_dir / "data_stamp.csv").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 2 and parts[1].strip() == 'ouster':
            ts_list.append(int(parts[0]))
    return np.array(ts_list, dtype=np.int64)


def load_sequence(seq: str, max_frames: int = None):
    """Load MulRan sequence → (frames, gt_timestamps, gt_poses, scan_timestamps)."""
    seq_dir  = MULRAN_ROOT / seq
    ouster_dir = seq_dir / "Ouster"
    gt_path    = seq_dir / "global_pose.csv"

    # MulRan: scan files named by timestamp
    bins = sorted(ouster_dir.glob("*.bin"))
    if max_frames:
        bins = bins[:max_frames]

    print(f"[MulRan/{seq}/Ouster] Loading {len(bins)} frames...")
    frames = []
    scan_timestamps = []
    for bf in bins:
        pts = load_mulran_frame(bf)
        frames.append(pts)
        scan_timestamps.append(int(bf.stem))

    gt_ts, gt_poses = load_mulran_gt(gt_path)
    scan_ts = np.array(scan_timestamps, dtype=np.int64)
    print(f"  GT poses: {len(gt_poses)}  scan pts avg: {np.mean([len(f) for f in frames]):.0f}")
    return frames, gt_ts, gt_poses, scan_ts


# ── ATE computation ────────────────────────────────────────────────────────────

def compute_ate(est_poses, est_timestamps_ns, gt_timestamps_ns, gt_poses):
    """ATE RMSE (Umeyama alignment) with nearest-timestamp matching."""
    est_ts = np.array(est_timestamps_ns, dtype=np.int64)
    matched_est, matched_gt = [], []
    for i, ts in enumerate(est_ts):
        idx = np.argmin(np.abs(gt_timestamps_ns - ts))
        dt_ms = abs(int(gt_timestamps_ns[idx]) - int(ts)) / 1e6
        if dt_ms < 200:
            matched_est.append(est_poses[i][:3, 3])
            matched_gt.append(gt_poses[idx][:3, 3])
    if len(matched_est) < 5:
        return float('nan')
    P = np.array(matched_est).T
    Q = np.array(matched_gt).T
    mu_p = P.mean(axis=1, keepdims=True)
    mu_q = Q.mean(axis=1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    errs = np.linalg.norm(R @ P + t - Q, axis=0)
    return float(np.sqrt(np.mean(errs**2)))


# ── Method runners ─────────────────────────────────────────────────────────────

def run_iv_gicp(frames, scan_ts, alpha, label, device, voxel_size=1.0,
                max_map_frames=10, window_size=1, **kw):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=voxel_size,
        source_voxel_size=0.3,
        alpha=alpha,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        min_motion_th=0.05,
        max_iterations=30,
        max_map_frames=max_map_frames,
        map_radius=80.0,              # spatial eviction: keep map near robot
        adaptive_voxelization=False,
        device=device,
        window_size=window_size,
        use_distribution_propagation=False,
        **kw,
    )
    poses, times = [], []
    print(f"\n[{label}] {len(frames)} frames  α={alpha}  voxel={voxel_size}  W={window_size}")
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(f[:, :3], f[:, 3], timestamp=float(scan_ts[i]))
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(result.pose.copy())
        if (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{len(frames)}  {np.mean(times[-50:]):.1f}ms/fr")
    hz = 1000.0 / np.mean(times[5:])
    print(f"  mean={np.mean(times[5:]):.1f}ms  {hz:.1f}Hz")
    return poses, times


def run_kiss_icp(frames, scan_ts, voxel_size=1.0):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.max_range = 80.0
    cfg.data.min_range = 0.5
    cfg.data.deskew = False
    cfg.mapping.voxel_size = voxel_size
    od = KissICP(config=cfg)
    poses, times = [], []
    print(f"\n[KISS-ICP] {len(frames)} frames  voxel={voxel_size}")
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        # Use sequential timestamps (nanosecond timestamps cause Sophus crash)
        od.register_frame(f[:, :3].astype(np.float64),
                          np.full(len(f), float(i)))
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(od.last_pose.copy())
        if (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{len(frames)}  {np.mean(times[-50:]):.1f}ms/fr")
    print(f"  mean={np.mean(times[5:]):.1f}ms  {1000/np.mean(times[5:]):.1f}Hz")
    return poses, times


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",        default="DCC01", choices=["DCC01"])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--alpha",      type=float, default=0.1)
    ap.add_argument("--voxel-size", type=float, default=1.0)
    ap.add_argument("--max-map-frames", type=int, default=10)
    ap.add_argument("--window-size", type=int, default=1)
    ap.add_argument("--skip-kiss",  action="store_true")
    args = ap.parse_args()

    frames, gt_ts, gt_poses, scan_ts = load_sequence(args.seq, args.max_frames)

    results = {}
    out_dir = RESULTS_ROOT / args.seq
    out_dir.mkdir(parents=True, exist_ok=True)

    # GICP-Baseline
    poses, _ = run_iv_gicp(frames, scan_ts, alpha=0.0, label="GICP-Baseline",
                            device=args.device, voxel_size=args.voxel_size,
                            max_map_frames=args.max_map_frames, window_size=args.window_size)
    ate = compute_ate(poses, scan_ts, gt_ts, gt_poses)
    results["gicp_baseline"] = ate
    print(f"  ATE RMSE: {ate:.4f}m")

    # IV-GICP
    poses, _ = run_iv_gicp(frames, scan_ts, alpha=args.alpha, label=f"IV-GICP (α={args.alpha})",
                            device=args.device, voxel_size=args.voxel_size,
                            max_map_frames=args.max_map_frames, window_size=args.window_size)
    ate = compute_ate(poses, scan_ts, gt_ts, gt_poses)
    results[f"iv_gicp_a{args.alpha}"] = ate
    print(f"  ATE RMSE: {ate:.4f}m")

    # KISS-ICP
    if not args.skip_kiss:
        poses, _ = run_kiss_icp(frames, scan_ts, voxel_size=args.voxel_size)
        ate = compute_ate(poses, scan_ts, gt_ts, gt_poses)
        results["kiss_icp"] = ate
        print(f"  ATE RMSE: {ate:.4f}m")

    # Summary
    print(f"\n{'='*55}")
    print(f"MulRan/{args.seq}  {args.max_frames}fr  voxel={args.voxel_size}  W={args.window_size}")
    print(f"{'Method':<25} {'ATE(m)':>8}")
    print(f"{'-'*35}")
    for k, v in results.items():
        print(f"  {k:<23} {v:>8.4f}m")
    if "kiss_icp" in results:
        best_ours = min(v for k, v in results.items() if k != "kiss_icp")
        delta = (best_ours - results["kiss_icp"]) / results["kiss_icp"] * 100
        marker = "✓" if delta < 0 else "✗"
        print(f"\n  vs KISS-ICP: {delta:+.1f}%  {marker}")

    import json
    (out_dir / f"results_{args.max_frames}fr.json").write_text(
        json.dumps({"seq": args.seq, "frames": args.max_frames,
                    "params": vars(args), "ate": results}, indent=2)
    )
    print(f"\nSaved → {out_dir}/results_{args.max_frames}fr.json")


if __name__ == "__main__":
    main()
