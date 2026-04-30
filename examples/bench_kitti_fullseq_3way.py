#!/usr/bin/env python3
"""
KITTI seq 00-10 full-seq 3-way benchmark (IV-GICP vs KISS-ICP vs GenZ-ICP).

Computes:
  - ATE RMSE (Umeyama-aligned, meters)           — standard SLAM metric
  - KITTI translation error (%) at 100-800m      — KITTI benchmark standard
  - RPE RMSE (translation, meters, delta=10fr)   — short-horizon accuracy

Motivation: KISS-ICP and GenZ-ICP papers report KITTI-standard RPE%. We add
this to match, while keeping ATE for cross-dataset consistency.

Saves results/kitti_fullseq/seq{XX}/results.json + summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_kitti_benchmark import (
    load_kitti_sequence,
    kitti_gt_to_lidar_frame,
    run_kiss_icp,
    run_genz_icp,
    ate_rmse,
    kitti_t_err,
    rpe_rmse,
)

OUT_ROOT = Path(__file__).parent.parent / "results" / "kitti_fullseq"


def run_iv_gicp_full(frames, device="cpu"):
    from iv_gicp.pipeline import IVGICPPipeline
    p = IVGICPPipeline(
        voxel_size=1.0,
        source_voxel_size=0.3,
        min_points_per_voxel=3,
        alpha=0.1,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        min_motion_th=0.5,
        max_map_points=100_000,
        max_map_frames=500,
        max_iterations=12,
        use_fim_weight=False,
        fim_auto_gate=0.0,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        max_source_points=0,
        device=device,
    )
    poses = []
    times = []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        r = p.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(r.pose.copy())
        if i % 200 == 0:
            print(f"  IV  {i}/{len(frames)}  {np.mean(times[-20:]):.0f}ms", end="\r")
    print(f"  IV  done  mean={np.mean(times):.1f}ms       ")
    return poses, times


def evaluate(poses, gt, label):
    if poses is None or not poses:
        return {"ate": None, "kitti_t_err": None, "rpe_t": None}
    ate, _ = ate_rmse(poses, gt)
    kt = kitti_t_err(poses, gt)
    rpe_t, _ = rpe_rmse(poses, gt, delta=10)
    print(f"  {label}  ATE={ate:.3f}m  KITTI_t={kt:.3f}%  RPE10={rpe_t:.3f}m")
    return {"ate": ate, "kitti_t_err": kt, "rpe_t": rpe_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=[f"{i:02d}" for i in range(11)])
    ap.add_argument("--skip-iv", action="store_true")
    ap.add_argument("--skip-kiss", action="store_true")
    ap.add_argument("--skip-genz", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for seq in args.seqs:
        print(f"\n{'='*72}\n[KITTI seq{seq}] loading full sequence")
        t0 = time.perf_counter()
        frames, poses_gt = load_kitti_sequence(seq, max_frames=None)
        print(f"  {len(frames)} frames loaded in {time.perf_counter()-t0:.1f}s")
        if not poses_gt:
            print("  (no GT)")
            continue
        poses_gt_lidar = kitti_gt_to_lidar_frame(poses_gt, seq)[:len(frames)]

        out = OUT_ROOT / f"seq{seq}"
        out.mkdir(parents=True, exist_ok=True)
        seq_res = {"n_frames": len(frames)}

        if not args.skip_kiss:
            t0 = time.perf_counter()
            ki_poses, ki_times = run_kiss_icp(frames)
            kiss_ms = float(np.mean(ki_times))
            seq_res["kiss"] = {
                **evaluate(ki_poses, poses_gt_lidar, "KISS"),
                "ms_mean": kiss_ms,
            }

        if not args.skip_iv:
            iv_poses, iv_times = run_iv_gicp_full(frames, device=args.device)
            iv_ms = float(np.mean(iv_times))
            seq_res["iv"] = {
                **evaluate(iv_poses, poses_gt_lidar, "IV"),
                "ms_mean": iv_ms,
            }

        if not args.skip_genz:
            gz_poses, _ = run_genz_icp(seq, max_frames=None, out_dir=out)
            if gz_poses:
                seq_res["genz"] = evaluate(gz_poses, poses_gt_lidar, "GenZ")

        (out / "results.json").write_text(json.dumps(seq_res, indent=2))
        all_results[f"seq{seq}"] = seq_res

    # Summary
    print(f"\n{'='*72}\n  KITTI FULL-SEQUENCE 3-WAY SUMMARY")
    print(f"{'Seq':<6} {'n':>5}   {'KISS_t%':>8} {'IV_t%':>8} {'GenZ_t%':>8}   "
          f"{'KISS_ATE':>8} {'IV_ATE':>8} {'GenZ_ATE':>8}")
    for k, r in all_results.items():
        n = r.get("n_frames", 0)
        ki_t = r.get("kiss", {}).get("kitti_t_err")
        iv_t = r.get("iv", {}).get("kitti_t_err")
        gz_t = r.get("genz", {}).get("kitti_t_err")
        ki_a = r.get("kiss", {}).get("ate")
        iv_a = r.get("iv", {}).get("ate")
        gz_a = r.get("genz", {}).get("ate")
        def fmt(v, w=8, p=3):
            return f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}"
        print(f"{k:<6} {n:>5}   {fmt(ki_t)} {fmt(iv_t)} {fmt(gz_t)}   "
              f"{fmt(ki_a)} {fmt(iv_a)} {fmt(gz_a)}")

    # Averages
    def avg(key, m):
        vals = [r.get(m, {}).get(key) for r in all_results.values()
                if r.get(m, {}).get(key) is not None]
        return float(np.mean(vals)) if vals else None

    print(f"\n{'avg':<6} {'':>5}   "
          f"{avg('kitti_t_err','kiss') or float('nan'):>8.3f} "
          f"{avg('kitti_t_err','iv') or float('nan'):>8.3f} "
          f"{avg('kitti_t_err','genz') or float('nan'):>8.3f}   "
          f"{avg('ate','kiss') or float('nan'):>8.3f} "
          f"{avg('ate','iv') or float('nan'):>8.3f} "
          f"{avg('ate','genz') or float('nan'):>8.3f}")

    (OUT_ROOT / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {OUT_ROOT}/summary.json")


if __name__ == "__main__":
    main()
