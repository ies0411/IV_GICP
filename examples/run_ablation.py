#!/usr/bin/env python3
"""
IV-GICP Ablation Study: 5 configurations on KITTI data.

Matches paper Table III (ablation):
  A: GICP baseline       (no adaptive, no intensity, no dist.prop.)
  B: + Adaptive only      (adaptive voxels, no intensity)
  C: + Intensity only     (fixed voxels, intensity constraints)
  D: IV-GICP w/o DP       (adaptive + intensity, no dist.prop.)
  E: Full IV-GICP         (adaptive + intensity + dist.prop.)

Usage:
  uv run python examples/run_ablation.py
  uv run python examples/run_ablation.py --data data/kitti/sample --max-frames 50
  uv run python examples/run_ablation.py -o output/ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from iv_gicp import IVGICPPipeline
from iv_gicp.kitti_loader import load_kitti_sequence
from iv_gicp.metrics import compute_ate, compute_rpe, compute_rpe_kitti


# Ablation configurations
# entropy_threshold/intensity_var_threshold set to 1e10 = effectively disabled (no splitting)
ABLATION_CONFIGS = {
    "A_gicp_baseline": {
        "label": "GICP (geometry-only, fixed voxels)",
        "alpha": 0.0,
        "entropy_threshold": 1e10,
        "intensity_var_threshold": 1e10,
        "use_distribution_propagation": False,
    },
    "B_adaptive_only": {
        "label": "+ Adaptive Voxelization",
        "alpha": 0.0,
        "entropy_threshold": 0.5,
        "intensity_var_threshold": 100.0,
        "use_distribution_propagation": False,
    },
    "C_intensity_only": {
        "label": "+ Intensity (fixed voxels)",
        "alpha": 0.1,
        "entropy_threshold": 1e10,
        "intensity_var_threshold": 1e10,
        "use_distribution_propagation": False,
    },
    "D_iv_gicp_no_dp": {
        "label": "IV-GICP w/o Dist. Prop.",
        "alpha": 0.1,
        "entropy_threshold": 0.5,
        "intensity_var_threshold": 0.01,
        "use_distribution_propagation": False,
    },
    "E_iv_gicp_full": {
        "label": "Full IV-GICP",
        "alpha": 0.1,
        "entropy_threshold": 0.5,
        "intensity_var_threshold": 0.01,
        "use_distribution_propagation": True,
    },
}


def run_one_config(
    config_name: str,
    config: dict,
    frames: list,
    poses_gt: list,
    common_params: dict,
) -> dict:
    """Run a single ablation configuration and return metrics."""
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  {config['label']}")
    print(f"{'='*60}")

    pipeline_params = {
        **common_params,
        "alpha": config["alpha"],
        "entropy_threshold": config["entropy_threshold"],
        "intensity_var_threshold": config["intensity_var_threshold"],
        "use_distribution_propagation": config["use_distribution_propagation"],
    }
    pipeline = IVGICPPipeline(**pipeline_params)

    t_start = time.time()
    frame_times = []

    for i, frame in enumerate(frames):
        t0 = time.time()
        result = pipeline.process_frame(frame, timestamp=float(i))
        frame_times.append(time.time() - t0)

        if i == 0:
            print(f"  Frame {i}: Map built, {pipeline.adaptive_map.get_voxel_count()} voxels")
        elif i % 20 == 0 or i == len(frames) - 1:
            t = result.pose[:3, 3]
            print(f"  Frame {i}: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]  ({frame_times[-1]:.2f}s)")

    total_time = time.time() - t_start
    traj = pipeline.get_trajectory()

    # Evaluate
    metrics = {"config": config_name, "label": config["label"]}
    if poses_gt is not None and len(poses_gt) == len(traj.poses):
        ate_rmse, ate_mean, _ = compute_ate(traj.poses, poses_gt, align=True)
        rpe_rmse, rpe_mean = compute_rpe(traj.poses, poses_gt)
        kitti_m = compute_rpe_kitti(traj.poses, poses_gt)
        metrics.update({
            "ate_rmse": round(ate_rmse, 4),
            "ate_mean": round(ate_mean, 4),
            "rpe_rmse": round(rpe_rmse, 4),
            "rpe_mean": round(rpe_mean, 4),
            "kitti_t_err_pct": round(kitti_m.get("t_err_pct", 0.0), 4),
            "kitti_r_err_deg_m": round(kitti_m.get("r_err_deg_m", 0.0), 6),
        })
    metrics.update({
        "total_time_s": round(total_time, 2),
        "avg_frame_time_s": round(np.mean(frame_times), 3),
        "n_frames": len(frames),
        "n_voxels_final": pipeline.adaptive_map.get_voxel_count(),
    })

    print(f"\n  ATE RMSE: {metrics.get('ate_rmse', 'N/A')} m")
    print(f"  RPE RMSE: {metrics.get('rpe_rmse', 'N/A')} m")
    print(f"  Time:     {total_time:.1f}s ({np.mean(frame_times):.2f}s/frame)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="IV-GICP Ablation Study")
    parser.add_argument("--data", type=str, default="data/kitti/sample")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("-o", "--output", type=str, default="output/ablation")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    print(f"Loading KITTI from {data_path}...")
    frames, poses_gt = load_kitti_sequence(str(data_path), args.max_frames, downsample=args.downsample)
    print(f"  {len(frames)} frames loaded")

    # Common pipeline parameters (shared across all configs)
    common_params = {
        "voxel_size": 1.0,
        "max_correspondence_distance": 5.0,
        "max_range": 80.0,
        "min_range": 2.0,
        "source_voxel_size": 0.3,
        "max_map_points": 200_000,
    }

    results = {}
    for name, config in ABLATION_CONFIGS.items():
        results[name] = run_one_config(name, config, frames, poses_gt, common_params)

    # Summary table
    print(f"\n{'='*80}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Adaptive':>8} {'Intens.':>8} {'D.P.':>5} {'ATE(m)':>8} {'RPE(m)':>8} {'Time(s)':>8}")
    print("-" * 80)

    adaptive_flags = [False, True, False, True, True]
    intensity_flags = [False, False, True, True, True]
    dp_flags = [False, False, False, False, True]

    for i, (name, m) in enumerate(results.items()):
        ada = "Y" if adaptive_flags[i] else "-"
        int_ = "Y" if intensity_flags[i] else "-"
        dp = "Y" if dp_flags[i] else "-"
        ate = f"{m.get('ate_rmse', 'N/A'):>8}" if isinstance(m.get('ate_rmse'), float) else "   N/A  "
        rpe = f"{m.get('rpe_rmse', 'N/A'):>8}" if isinstance(m.get('rpe_rmse'), float) else "   N/A  "
        tm = f"{m['total_time_s']:>8.1f}"
        print(f"{m['label']:<25} {ada:>8} {int_:>8} {dp:>5} {ate} {rpe} {tm}")

    # Save results
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
