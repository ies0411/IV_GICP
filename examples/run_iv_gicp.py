#!/usr/bin/env python3
"""
IV-GICP 예제: 포인트 클라우드 데이터로 오도메트리 실행

Usage:
  python examples/run_iv_gicp.py                  # 랜덤 데이터로 테스트
  python examples/run_iv_gicp.py --pcd <file.pcd>  # PCD 파일 로드
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from typing import Optional

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iv_gicp import IVGICPPipeline, AdaptiveVoxelMap, IVGICP


def generate_synthetic_sequence(
    n_frames: int = 20,
    points_per_frame: int = 2000,
    seed: Optional[int] = None,
) -> tuple:
    """간단한 합성 궤적 + 포인트 클라우드 시퀀스 생성.
    각 프레임의 포인트는 해당 프레임의 로컬(센서) 좌표계에 있음.
    Returns: (frames, poses_gt)
    """
    if seed is not None:
        np.random.seed(seed)
    frames = []
    poses_gt = []
    for i in range(n_frames):
        t = i * 0.5
        pose = np.eye(4)
        pose[0, 3] = t * 2
        pose[1, 3] = 0.1 * np.sin(t * 0.5)
        R = pose[:3, :3]
        poses_gt.append(pose.copy())

        pts_local = np.random.randn(points_per_frame, 3) * 3
        pts_local[:, 0] += 5
        intensity = np.random.rand(points_per_frame) * 100 + 50
        frames.append(np.column_stack([pts_local, intensity]))

    return frames, poses_gt


def load_pcd(path: str) -> np.ndarray:
    """PCD 파일 로드 (Open3D 사용)."""
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if hasattr(pcd, "colors") and np.asarray(pcd.colors).size > 0:
            # Grayscale intensity from RGB
            col = np.asarray(pcd.colors)
            intensity = np.mean(col, axis=1) * 255
        else:
            intensity = np.zeros(len(pts))
        return np.column_stack([pts, intensity])
    except ImportError:
        raise ImportError("Open3D required for PCD: pip install open3d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", type=str, help="PCD file path")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default="output",
        metavar="DIR",
        help="Save figures to directory (default: output/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Figure format: png or pdf (LaTeX용)",
    )
    parser.add_argument(
        "--no-open3d-3d",
        action="store_true",
        help="Open3D 3D 대신 matplotlib 사용 (headless에서 Open3D 실패 시)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Synthetic data random seed (reproducibility)",
    )
    parser.add_argument(
        "--load-data",
        type=str,
        default=None,
        metavar="DIR",
        help="Load saved synthetic sequence from directory",
    )
    args = parser.parse_args()

    poses_gt = None
    if args.pcd:
        print(f"Loading {args.pcd}...")
        points = load_pcd(args.pcd)
        frames = [points]
    elif args.load_data:
        from iv_gicp.data_io import load_synthetic_sequence

        frames, poses_gt = load_synthetic_sequence(args.load_data)
        print(f"Loaded {len(frames)} frames from {args.load_data}")
    else:
        print("Generating synthetic sequence...")
        frames, poses_gt = generate_synthetic_sequence(args.frames, args.points, seed=args.seed)

    pipeline = IVGICPPipeline(
        voxel_size=0.5,
        entropy_threshold=2.0,
        intensity_var_threshold=50.0,
        alpha=0.02,
        max_correspondence_distance=1.5,
    )

    print("Running IV-GICP pipeline...")
    for i, frame in enumerate(frames):
        result = pipeline.process_frame(frame, timestamp=float(i))
        if i == 0:
            print(f"  Frame {i}: Initial map built, {pipeline.adaptive_map.get_voxel_count()} voxels")
        else:
            t = result.pose[:3, 3]
            print(f"  Frame {i}: pose = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

    traj = pipeline.get_trajectory()
    print(f"\nTrajectory: {len(traj.poses)} poses")

    # GT 비교 (synthetic 데이터인 경우)
    metrics_dict = None
    if poses_gt is not None and len(poses_gt) == len(traj.poses):
        from iv_gicp.metrics import compute_ate, compute_rpe

        ate_rmse, ate_mean, ate_per_frame = compute_ate(traj.poses, poses_gt)
        rpe_rmse, rpe_mean = compute_rpe(traj.poses, poses_gt)
        metrics_dict = {
            "ate_rmse": ate_rmse,
            "ate_mean": ate_mean,
            "rpe_rmse": rpe_rmse,
            "rpe_mean": rpe_mean,
        }
        print(f"\n--- GT 비교 ---")
        print(f"  ATE RMSE: {ate_rmse:.4f} m")
        print(f"  ATE Mean: {ate_mean:.4f} m")
        print(f"  RPE RMSE: {rpe_rmse:.4f} m")
        print(f"  RPE Mean: {rpe_mean:.4f} m")

    # Synthetic 데이터 + Figure + Metrics 저장
    if not args.no_save:
        out_path = Path(args.save)
        if poses_gt is not None and not args.pcd and not args.load_data:
            from iv_gicp.data_io import save_synthetic_sequence, save_synthetic_sequence_pcd

            data_dir = str(out_path / "data")
            save_synthetic_sequence(frames, poses_gt, data_dir)
            save_synthetic_sequence_pcd(frames, poses_gt, data_dir)
        if metrics_dict:
            import json

            with open(out_path / "metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"Saved: {out_path / 'metrics.json'}")
        from iv_gicp.visualize import save_all_figures

        save_all_figures(
            pipeline,
            args.save,
            fmt=args.format,
            use_open3d_3d=not args.no_open3d_3d,
            poses_gt=poses_gt,
        )

    print("Done.")


if __name__ == "__main__":
    main()
