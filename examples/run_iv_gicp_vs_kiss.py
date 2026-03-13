#!/usr/bin/env python3
"""
IV-GICP Full (C1+C2+C3) vs KISS-ICP on all datasets, 50 frames each.

Usage:
  uv run python examples/run_iv_gicp_vs_kiss.py
  uv run python examples/run_iv_gicp_vs_kiss.py --device cuda
  uv run python examples/run_iv_gicp_vs_kiss.py --datasets kitti subt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

# Reuse data loaders and metrics from run_ablation
import run_ablation as ra

MAX_FRAMES = 50


def run_iv_gicp_full_subprocess(dataset_key, max_frames, device):
    """Run IV-GICP Full in subprocess; return (ate, rpe, ms) or (nan, nan, nan) on failure."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset", dataset_key,
        "--method", "iv_gicp",
        "--max-frames", str(max_frames),
        "--device", device,
    ]
    try:
        out = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode != 0:
            return float("nan"), float("nan"), float("nan")
        for line in out.stdout.strip().split("\n"):
            if line.startswith("ATE="):
                parts = line.split()
                ate = float(parts[0].split("=")[1])
                rpe = float(parts[1].split("=")[1])
                ms = float(parts[2].split("=")[1])
                return ate, rpe, ms
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return float("nan"), float("nan"), float("nan")


def run_kiss_icp_subprocess(dataset_key, max_frames):
    """Run KISS-ICP in subprocess; return (ate, rpe, ms) or (nan, nan, nan) on failure."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset", dataset_key,
        "--method", "kiss",
        "--max-frames", str(max_frames),
    ]
    try:
        out = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if out.returncode != 0:
            return float("nan"), float("nan"), float("nan")
        for line in out.stdout.strip().split("\n"):
            if line.startswith("ATE=") and " " in line and "RPE=" in line:
                parts = line.split()
                ate = float(parts[0].split("=")[1])
                rpe = float(parts[1].split("=")[1])
                ms = float(parts[2].split("=")[1])
                return ate, rpe, ms
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return float("nan"), float("nan"), float("nan")


def run_worker(dataset_key, method, max_frames, device="auto"):
    """Worker mode: load dataset, run one method, print ATE= RPE= MS= and exit."""
    ds = ra.DATASETS[dataset_key]
    frames, poses_gt = ds["loader"](max_frames=max_frames, **ds["loader_kw"])
    if len(frames) < 2 or len(poses_gt) < 2:
        print("ATE=nan RPE=nan MS=nan")
        return
    if method == "iv_gicp":
        ate, rpe, hz, _ = ra.run_config(
            "F: Full(C1+2+3)",
            frames,
            poses_gt,
            alpha=ds["alpha"],
            use_fim_weight=True,
            use_entropy_alpha=True,
            window=1,
            params=ds["params"],
            device=device,
            auto_alpha=True,
        )
        ms = 1000.0 / hz if hz and hz > 0 else 0.0
    else:
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.config import KISSConfig
        cfg = KISSConfig()
        cfg.data.max_range = 80.0
        cfg.data.min_range = 0.5
        cfg.data.deskew = False
        cfg.mapping.voxel_size = 1.0
        kiss = KissICP(config=cfg)
        poses, times_ms = [], []
        for i, f in enumerate(frames):
            pts = np.asarray(f[:, :3], dtype=np.float64)
            # Valid points only (no NaN/Inf); KISS-ICP can crash on degenerate input
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]
            if pts.shape[0] < 100:
                # Too few points: keep identity pose to avoid Sophus SO3::exp failure
                poses.append(poses[-1].copy() if poses else np.eye(4))
                times_ms.append(0.0)
                continue
            # Per-frame timestamp (same as run_kitti_benchmark / run_geode_eval)
            timestamps = np.full(pts.shape[0], float(i), dtype=np.float64)
            t0 = time.perf_counter()
            kiss.register_frame(pts, timestamps)
            times_ms.append((time.perf_counter() - t0) * 1000)
            poses.append(kiss.last_pose.copy())
        ate = ra.ate_rmse(poses, poses_gt)
        rpe = ra.rpe_rmse(poses, poses_gt, delta=1)
        ms = np.mean(times_ms) if times_ms else 0.0
    print(f"ATE={ate:.6f} RPE={rpe:.6f} MS={ms:.2f}")


def main():
    parser = argparse.ArgumentParser(description="IV-GICP Full vs KISS-ICP (50 fr per dataset)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--datasets", nargs="+", default=["kitti", "subt", "metro", "geode"], help="Dataset keys to run")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.dataset, args.method, args.max_frames, args.device)
        return

    keys = [k for k in args.datasets if k in ra.DATASETS]
    results = []

    for k in keys:
        ds = ra.DATASETS[k]
        print(f"\n[{k}] {ds['label']}  ({args.max_frames} frames)")
        print("  IV-GICP Full...", end=" ", flush=True)
        ate_iv, rpe_iv, ms_iv = run_iv_gicp_full_subprocess(k, args.max_frames, args.device)
        print(f"ATE={ate_iv:.4f}m  RPE={rpe_iv:.4f}m  {ms_iv:.0f}ms/fr")
        print("  KISS-ICP...", end=" ", flush=True)
        ate_kiss, rpe_kiss, ms_kiss = run_kiss_icp_subprocess(k, args.max_frames)
        print(f"ATE={ate_kiss:.4f}m  RPE={rpe_kiss:.4f}m  {ms_kiss:.0f}ms/fr")
        results.append((k, ds["label"], ate_iv, rpe_iv, ms_iv, ate_kiss, rpe_kiss, ms_kiss))

    print("\n" + "=" * 80)
    print("  IV-GICP Full vs KISS-ICP  (50 frames per dataset)")
    print("=" * 80)
    print(f"  {'Dataset':<28}  {'IV-GICP ATE':>10}  {'KISS ATE':>10}  {'Winner':<8}  {'IV-GICP RPE':>10}  {'KISS RPE':>10}")
    print("  " + "-" * 76)
    for k, label, ate_iv, rpe_iv, ms_iv, ate_kiss, rpe_kiss, ms_kiss in results:
        if np.isnan(ate_iv):
            print(f"  {label:<28}  {'—':>10}  {'—':>10}  skip")
            continue
        if np.isnan(ate_kiss):
            winner = "IV-GICP (KISS n/a)"
        else:
            winner = "IV-GICP" if ate_iv < ate_kiss else "KISS-ICP"
        print(f"  {label:<28}  {ate_iv:>10.4f}  {ate_kiss:>10.4f}  {winner:<8}  {rpe_iv:>10.4f}  {rpe_kiss:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
