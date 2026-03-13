#!/usr/bin/env python3
"""
KITTI Data로 IV-GICP 실행 및 종합 리포트 저장

Supports both KITTI raw format and KITTI odometry benchmark format:
  Raw:      data_root/velodyne_points/data/*.bin + oxts/data/*.txt
  Odometry: data_root/sequences/XX/velodyne/*.bin + poses/XX.txt

Usage:
  uv run python examples/run_kitti.py
  uv run python examples/run_kitti.py --data data/kitti/sample --max-frames 50
  uv run python examples/run_kitti.py --format odometry --data /path/to/kitti --seq 00
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from iv_gicp import IVGICPPipeline
from iv_gicp.kitti_loader import load_kitti_sequence, load_kitti_odometry_sequence
from iv_gicp.metrics import compute_ate, compute_rpe, compute_rpe_kitti


def main():
    parser = argparse.ArgumentParser(description="IV-GICP KITTI Evaluation")
    parser.add_argument(
        "--data",
        type=str,
        default="data/kitti/sample",
        help="KITTI data root directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["raw", "odometry"],
        default="raw",
        help="KITTI data format (raw: velodyne_points+oxts, odometry: sequences+poses)",
    )
    parser.add_argument("--seq", type=str, default="00", help="Sequence ID (odometry format)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--downsample", type=int, default=5, help="Point stride downsample")
    parser.add_argument("-o", "--output", type=str, default="output/kitti", help="Output directory")
    parser.add_argument("--no-open3d-3d", action="store_true", help="Use matplotlib for 3D")
    # Pipeline parameters
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-corr-dist", type=float, default=5.0)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--intensity-var-threshold", type=float, default=0.01)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    # Load data
    print(f"Loading KITTI ({args.format}) from {data_path}...")
    if args.format == "odometry":
        frames, poses_gt = load_kitti_odometry_sequence(
            str(data_path),
            args.seq,
            args.max_frames,
            downsample=args.downsample,
        )
    else:
        frames, poses_gt = load_kitti_sequence(
            str(data_path),
            args.max_frames,
            downsample=args.downsample,
        )

    print(f"  Loaded {len(frames)} frames")
    if poses_gt:
        print(f"  Ground truth: {len(poses_gt)} poses")
    else:
        print("  Ground truth: N/A")

    # Pipeline
    pipeline = IVGICPPipeline(
        voxel_size=args.voxel_size,
        entropy_threshold=args.entropy_threshold,
        intensity_var_threshold=args.intensity_var_threshold,
        alpha=args.alpha,
        max_correspondence_distance=args.max_corr_dist,
        max_range=80.0,
        min_range=2.0,
    )

    print("\nRunning IV-GICP...")
    frame_times = []
    for i, frame in enumerate(frames):
        t0 = time.time()
        result = pipeline.process_frame(frame, timestamp=float(i))
        dt = time.time() - t0
        frame_times.append(dt)

        if i == 0:
            if pipeline.flat_map is not None:
                n_voxels = len(pipeline.flat_map.leaves)
                print(f"  Frame {i}: Map built, {n_voxels} voxels ({dt:.2f}s)")
            elif getattr(pipeline, "_cpp_voxel_map", None) is not None:
                n_voxels = len(pipeline._cpp_voxel_map)
                print(f"  Frame {i}: Map built (C++), {n_voxels} voxels ({dt:.2f}s)")
            else:
                print(f"  Frame {i}: Map built ({dt:.2f}s)")
        elif i % 20 == 0 or i == len(frames) - 1:
            t = result.pose[:3, 3]
            print(f"  Frame {i}: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] ({dt:.2f}s)")

    traj = pipeline.get_trajectory()
    total_time = sum(frame_times)
    avg_time = np.mean(frame_times[1:]) if len(frame_times) > 1 else 0

    print(f"\nTrajectory: {len(traj.poses)} poses")
    print(f"Total time: {total_time:.1f}s, Avg: {avg_time:.3f}s/frame")

    # Evaluate
    metrics = {}
    if poses_gt is not None and len(poses_gt) == len(traj.poses):
        ate_rmse, ate_mean, per_frame = compute_ate(traj.poses, poses_gt, align=True)
        rpe_rmse, rpe_mean = compute_rpe(traj.poses, poses_gt)
        kitti_metrics = compute_rpe_kitti(traj.poses, poses_gt)
        metrics = {
            "ate_rmse": ate_rmse,
            "ate_mean": ate_mean,
            "rpe_rmse": rpe_rmse,
            "rpe_mean": rpe_mean,
            "kitti_t_err_pct": kitti_metrics.get("t_err_pct", 0.0),
            "kitti_r_err_deg_m": kitti_metrics.get("r_err_deg_m", 0.0),
        }
        print("\n--- Evaluation ---")
        print(f"  ATE RMSE:    {ate_rmse:.4f} m")
        print(f"  ATE Mean:    {ate_mean:.4f} m")
        print(f"  RPE RMSE:    {rpe_rmse:.4f} m")
        print(f"  RPE Mean:    {rpe_mean:.4f} m")
        print(f"  KITTI t_err: {kitti_metrics.get('t_err_pct', 0.0):.2f} %")
        print(f"  KITTI r_err: {kitti_metrics.get('r_err_deg_m', 0.0):.4f} deg/m")

    # Save outputs
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Full report
    report = {
        "dataset": f"KITTI {args.format}",
        "data_path": str(data_path),
        "sequence": args.seq if args.format == "odometry" else "N/A",
        "n_frames": len(frames),
        "has_gt": poses_gt is not None,
        "timestamp": datetime.now().isoformat(),
        "pipeline_params": {
            "voxel_size": args.voxel_size,
            "alpha": args.alpha,
            "entropy_threshold": args.entropy_threshold,
            "intensity_var_threshold": args.intensity_var_threshold,
            "max_correspondence_distance": args.max_corr_dist,
        },
        "timing": {
            "total_time_s": round(total_time, 2),
            "avg_frame_time_s": round(avg_time, 3),
            "frame_times": [round(t, 3) for t in frame_times],
        },
        "metrics": metrics,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if metrics:
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Save trajectory
    poses_arr = np.array(traj.poses)
    np.save(out / "poses_est.npy", poses_arr)

    # KITTI format (3x4 flattened per line)
    with open(out / "poses_est.txt", "w") as f:
        for T in traj.poses:
            vals = T[:3, :].flatten()
            f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")

    # Figures
    from iv_gicp.visualize import save_all_figures

    save_all_figures(
        pipeline,
        str(out),
        fmt="png",
        use_open3d_3d=not args.no_open3d_3d,
        poses_gt=poses_gt,
    )

    # Human-readable report
    with open(out / "report.txt", "w", encoding="utf-8") as f:
        f.write("IV-GICP KITTI Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data:       {data_path}\n")
        f.write(f"Format:     {args.format}\n")
        f.write(f"Frames:     {len(frames)}\n")
        f.write(f"GT:         {'Yes' if poses_gt else 'No'}\n")
        f.write(f"Time:       {total_time:.1f}s ({avg_time:.3f}s/frame)\n")
        if metrics:
            f.write(f"\nATE RMSE:   {metrics['ate_rmse']:.4f} m\n")
            f.write(f"ATE Mean:   {metrics['ate_mean']:.4f} m\n")
            f.write(f"RPE RMSE:   {metrics['rpe_rmse']:.4f} m\n")
            f.write(f"RPE Mean:   {metrics['rpe_mean']:.4f} m\n")
            f.write(f"KITTI t%%:   {metrics['kitti_t_err_pct']:.2f}\n")
            f.write(f"KITTI r:    {metrics['kitti_r_err_deg_m']:.4f} deg/m\n")
        f.write(f"\nOutput:     {out}/\n")

    print(f"\nSaved to {out}/")
    print("Done.")


if __name__ == "__main__":
    main()
