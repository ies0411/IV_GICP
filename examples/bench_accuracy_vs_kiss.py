#!/usr/bin/env python3
"""
IV-GICP vs KISS-ICP 정확도 비교 (합성 궤적, GT 있음).

동일한 합성 시퀀스에 대해 두 방법을 돌리고 ATE/RPE로 정확도가 유지되는지 확인.
데이터셋 없이 실행 가능.

Usage:
  uv run python examples/bench_accuracy_vs_kiss.py
  uv run python examples/bench_accuracy_vs_kiss.py --frames 40 --pts 3000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from iv_gicp.metrics import compute_ate, compute_rpe


def make_synthetic_sequence(n_frames: int, n_pts: int, seed: int = 42):
    """합성 궤적 GT + 매 프레임 스캔 (센서 좌표계)."""
    np.random.seed(seed)
    # World-frame point cloud (고정)
    pts_world = np.random.randn(n_pts, 3).astype(np.float64) * 3.0
    intensity = np.clip(np.abs(np.random.randn(n_pts).astype(np.float64)), 0.01, 1.0)
    pts_world_h = np.column_stack([pts_world, np.ones(n_pts)])  # (N, 4)

    # GT trajectory: 직선 + 약간의 yaw
    poses_gt = []
    for i in range(n_frames):
        T = np.eye(4)
        T[0, 3] = 0.15 * i
        T[1, 3] = 0.08 * i
        T[2, 3] = 0.02 * i
        angle = 0.01 * i
        c, s = np.cos(angle), np.sin(angle)
        T[:2, :2] = [[c, -s], [s, c]]
        poses_gt.append(T.copy())

    # 각 프레임 스캔: sensor frame = inv(T_gt) @ world
    frames_4d = []
    for T_gt in poses_gt:
        T_inv = np.linalg.inv(T_gt)
        pts_sensor = (T_inv @ pts_world_h.T).T[:, :3]
        frames_4d.append(np.column_stack([pts_sensor, intensity]))
    frames_xyz = [f[:, :3].astype(np.float64) for f in frames_4d]
    return frames_4d, frames_xyz, poses_gt


def main():
    p = argparse.ArgumentParser(description="IV-GICP vs KISS-ICP accuracy on synthetic trajectory")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--pts", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print("Generating synthetic trajectory (GT + scans)...")
    frames_4d, frames_xyz, poses_gt = make_synthetic_sequence(args.frames, args.pts, args.seed)
    print("  frames={}, pts/frame={}\n".format(args.frames, args.pts))

    # ── IV-GICP ─────────────────────────────────────────────────────────────
    from iv_gicp import IVGICPPipeline
    pipeline = IVGICPPipeline(device=args.device)
    poses_iv = []
    for i, pts in enumerate(frames_4d):
        pipeline.process_frame(pts, timestamp=float(i))
        poses_iv.append(pipeline.get_trajectory().poses[-1].copy())
    ate_iv, ate_mean_iv, _ = compute_ate(poses_iv, poses_gt, align=True)
    rpe_iv, rpe_mean_iv = compute_rpe(poses_iv, poses_gt, delta=1)

    # ── KISS-ICP ────────────────────────────────────────────────────────────
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.max_range = 80.0
    cfg.data.min_range = 2.0
    cfg.data.deskew = False
    cfg.mapping.voxel_size = 1.0
    kiss = KissICP(config=cfg)
    poses_kiss = []
    for pts in frames_xyz:
        kiss.register_frame(pts, np.zeros(len(pts)))
        poses_kiss.append(kiss.last_pose.copy())
    ate_kiss, ate_mean_kiss, _ = compute_ate(poses_kiss, poses_gt, align=True)
    rpe_kiss, rpe_mean_kiss = compute_rpe(poses_kiss, poses_gt, delta=1)

    # ── 결과 ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Accuracy vs KISS-ICP (synthetic trajectory with GT)")
    print("=" * 60)
    print("  Metric          |  IV-GICP   |  KISS-ICP  |  (lower is better)")
    print("  ATE RMSE (m)    |  {:>8.4f}  |  {:>8.4f}  |".format(ate_iv, ate_kiss))
    print("  ATE mean (m)    |  {:>8.4f}  |  {:>8.4f}  |".format(ate_mean_iv, ate_mean_kiss))
    print("  RPE RMSE (m)    |  {:>8.4f}  |  {:>8.4f}  |".format(rpe_iv, rpe_kiss))
    print("  RPE mean (m)    |  {:>8.4f}  |  {:>8.4f}  |".format(rpe_mean_iv, rpe_mean_kiss))
    print("=" * 60)
    if ate_iv <= ate_kiss * 1.5 and rpe_iv <= rpe_kiss * 1.5:
        print("  -> IV-GICP accuracy is comparable or better than KISS-ICP on this setup.")
    else:
        print("  -> KISS-ICP is more accurate on this synthetic; IV-GICP may need tuning.")


if __name__ == "__main__":
    main()
