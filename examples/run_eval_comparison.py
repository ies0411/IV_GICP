#!/usr/bin/env python3
"""
IV-GICP vs KISS-ICP vs GICP-baseline comparison on KITTI sample data.

Outputs eval.md with ATE/RPE metrics and timing.

Usage:
  uv run python examples/run_eval_comparison.py
  uv run python examples/run_eval_comparison.py --data data/kitti/sample --max-frames 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from iv_gicp import IVGICPPipeline
from iv_gicp.kitti_loader import load_kitti_sequence
from iv_gicp.metrics import compute_ate, compute_rpe, compute_rpe_kitti


# ──────────────────────────────────────────────────────────────────────────────
# KISS-ICP runner
# ──────────────────────────────────────────────────────────────────────────────
def run_kiss_icp(frames_xyz, poses_gt):
    """Run KISS-ICP on list of (N,3) point clouds. Returns metrics dict."""
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    config = KISSConfig()
    config.data.max_range = 80.0
    config.data.min_range = 2.0
    config.mapping.voxel_size = 1.0  # must be set; KissICP requires explicit voxel_size
    kiss = KissICP(config=config)

    poses = []
    frame_times = []
    for pts in frames_xyz:
        t0 = time.time()
        timestamps = np.zeros(len(pts))
        kiss.register_frame(pts, timestamps)
        frame_times.append(time.time() - t0)
        poses.append(kiss.last_pose.copy())

    return _eval(poses, poses_gt, frame_times, "KISS-ICP")


# ──────────────────────────────────────────────────────────────────────────────
# IV-GICP runner (shared helper)
# ──────────────────────────────────────────────────────────────────────────────
def run_iv_gicp(frames_xyzI, poses_gt, config_name, **kwargs):
    pipeline = IVGICPPipeline(**kwargs)
    poses = []
    frame_times = []
    for pts in frames_xyzI:
        t0 = time.time()
        pipeline.process_frame(pts)
        frame_times.append(time.time() - t0)
        poses.append(pipeline.get_trajectory().poses[-1].copy())
    return _eval(poses, poses_gt, frame_times, config_name)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def _eval(poses, poses_gt, frame_times, label):
    result = {"label": label, "n_frames": len(poses)}
    if poses_gt is not None and len(poses_gt) == len(poses):
        ate_rmse, ate_mean, _ = compute_ate(poses, poses_gt, align=True)
        rpe_rmse, rpe_mean = compute_rpe(poses, poses_gt)
        kitti_m = compute_rpe_kitti(poses, poses_gt)
        result.update({
            "ate_rmse": round(ate_rmse, 4),
            "ate_mean": round(ate_mean, 4),
            "rpe_rmse": round(rpe_rmse, 4),
            "rpe_mean": round(rpe_mean, 4),
            "kitti_t_err_pct": round(kitti_m.get("t_err_pct", float("nan")), 4),
            "kitti_r_err_deg_m": round(kitti_m.get("r_err_deg_m", float("nan")), 6),
        })
    result["avg_frame_ms"] = round(np.mean(frame_times) * 1000, 1)
    result["total_time_s"] = round(sum(frame_times), 2)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# eval.md writer
# ──────────────────────────────────────────────────────────────────────────────
def write_eval_md(results, data_path, n_frames, out_path):
    from datetime import date
    today = date.today().isoformat()

    lines = []
    lines.append("# IV-GICP Evaluation Report\n")
    lines.append(f"**Date:** {today}  ")
    lines.append(f"**Dataset:** KITTI Raw (`{data_path}`)  ")
    lines.append(f"**Frames:** {n_frames}  \n")

    lines.append("---\n")
    lines.append("## Methods\n")
    lines.append("| ID | Method | Adaptive Voxel | Intensity (4D) | Dist. Prop. |")
    lines.append("|---|---|:---:|:---:|:---:|")
    method_info = [
        ("GICP Baseline", "✗", "✗", "✗"),
        ("IV-GICP (Adaptive only)", "✓", "✗", "✗"),
        ("IV-GICP (Intensity only)", "✗", "✓", "✗"),
        ("IV-GICP (Full, no DP)", "✓", "✓", "✗"),
        ("IV-GICP (Full)", "✓", "✓", "✓"),
        ("KISS-ICP", "✓ (adaptive threshold)", "✗", "✗"),
    ]
    for i, (name, ada, inten, dp) in enumerate(method_info):
        lines.append(f"| {i+1} | {name} | {ada} | {inten} | {dp} |")

    lines.append("\n---\n")
    lines.append("## Results: Trajectory Accuracy\n")
    lines.append("| Method | ATE RMSE (m) | ATE Mean (m) | RPE RMSE (m) | RPE Mean (m) | KITTI t-err (%) | KITTI r-err (°/m) |")
    lines.append("|---|---|---|---|---|---|---|")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for r in results:
        row = (
            f"| **{r['label']}** "
            f"| {fmt(r.get('ate_rmse', 'N/A'))} "
            f"| {fmt(r.get('ate_mean', 'N/A'))} "
            f"| {fmt(r.get('rpe_rmse', 'N/A'))} "
            f"| {fmt(r.get('rpe_mean', 'N/A'))} "
            f"| {fmt(r.get('kitti_t_err_pct', 'N/A'))} "
            f"| {fmt(r.get('kitti_r_err_deg_m', 'N/A'))} |"
        )
        lines.append(row)

    lines.append("\n---\n")
    lines.append("## Results: Runtime\n")
    lines.append("| Method | Avg Frame Time (ms) | Total Time (s) | Frames |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(
            f"| **{r['label']}** | {r['avg_frame_ms']} | {r['total_time_s']} | {r['n_frames']} |"
        )

    lines.append("\n---\n")
    lines.append("## Notes\n")
    lines.append("- **ATE (Absolute Trajectory Error):** 전체 궤적의 절대 위치 오차. `align=True`로 Umeyama alignment 후 계산.")
    lines.append("- **RPE (Relative Pose Error):** 연속 프레임 간 상대 포즈 오차.")
    lines.append("- **KITTI t-err:** KITTI 공식 평가 지표 (경로 길이 대비 번역 오차 %).")
    lines.append("- **KISS-ICP:** geometry-only, adaptive voxel threshold. Intensity 미사용.")
    lines.append("- **Dist. Prop.:** Retroactive distribution propagation (Lie theory 기반 복셀 통계 업데이트).")
    lines.append("- 모든 실험은 동일 KITTI Raw 시퀀스, 동일 다운샘플링 조건으로 수행.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kitti/sample")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("-o", "--output", default="output/eval")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    print(f"Loading KITTI from {data_path} ...")
    frames_xyzI, poses_gt = load_kitti_sequence(
        str(data_path), args.max_frames, downsample=args.downsample
    )
    frames_xyz = [f[:, :3] for f in frames_xyzI]
    n = len(frames_xyzI)
    print(f"  {n} frames loaded\n")

    common = dict(
        voxel_size=1.0,
        max_correspondence_distance=5.0,
        max_range=80.0,
        min_range=2.0,
        source_voxel_size=0.3,
        max_map_points=200_000,
    )

    results = []

    print("=== [1/6] GICP Baseline ===")
    results.append(run_iv_gicp(frames_xyzI, poses_gt, "GICP Baseline",
        alpha=0.0,
        entropy_threshold=1e10,
        intensity_var_threshold=1e10,
        use_distribution_propagation=False,
        **common,
    ))

    print("=== [2/6] IV-GICP (Adaptive only) ===")
    results.append(run_iv_gicp(frames_xyzI, poses_gt, "IV-GICP (Adaptive only)",
        alpha=0.0,
        entropy_threshold=0.5,
        intensity_var_threshold=100.0,
        use_distribution_propagation=False,
        **common,
    ))

    print("=== [3/6] IV-GICP (Intensity only) ===")
    results.append(run_iv_gicp(frames_xyzI, poses_gt, "IV-GICP (Intensity only)",
        alpha=0.1,
        entropy_threshold=1e10,
        intensity_var_threshold=1e10,
        use_distribution_propagation=False,
        **common,
    ))

    print("=== [4/6] IV-GICP (Full, no DP) ===")
    results.append(run_iv_gicp(frames_xyzI, poses_gt, "IV-GICP (Full, no DP)",
        alpha=0.1,
        entropy_threshold=0.5,
        intensity_var_threshold=0.01,
        use_distribution_propagation=False,
        **common,
    ))

    print("=== [5/6] IV-GICP (Full) ===")
    results.append(run_iv_gicp(frames_xyzI, poses_gt, "IV-GICP (Full)",
        alpha=0.1,
        entropy_threshold=0.5,
        intensity_var_threshold=0.01,
        use_distribution_propagation=True,
        **common,
    ))

    print("=== [6/6] KISS-ICP ===")
    results.append(run_kiss_icp(frames_xyz, poses_gt))

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Method':<30} {'ATE(m)':>8} {'RPE(m)':>8} {'KITTI-t%':>9} {'ms/frame':>9}")
    print("-" * 90)
    for r in results:
        ate = f"{r.get('ate_rmse', float('nan')):.4f}"
        rpe = f"{r.get('rpe_rmse', float('nan')):.4f}"
        kt = f"{r.get('kitti_t_err_pct', float('nan')):.4f}"
        ms = f"{r['avg_frame_ms']:.1f}"
        print(f"{r['label']:<30} {ate:>8} {rpe:>8} {kt:>9} {ms:>9}")

    out = Path(args.output)
    write_eval_md(results, args.data, n, out / "eval.md")


if __name__ == "__main__":
    main()
