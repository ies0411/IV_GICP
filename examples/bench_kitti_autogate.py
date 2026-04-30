#!/usr/bin/env python3
"""
KITTI seq00 auto-gate regression check (full-seq).

Verifies that fim_auto_gate does not regress on well-conditioned outdoor data.

Configs:
  IV-GICP c1_off   (paper KITTI config: alpha=0.1, use_fim_weight=False)
  IV-GICP c1_on    (degrades KITTI in prior ablation: +79.4%)
  IV-GICP gate_XXX (auto-gate variants)

Expect: gate_* ≈ c1_off (not c1_on). Gap_ratio should stay above threshold on KITTI.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_ROOT = Path(__file__).parent.parent / "results" / "kitti" / "autogate"


def run_iv(frames, use_fim_weight, fim_auto_gate, label):
    from iv_gicp.pipeline import IVGICPPipeline
    p = IVGICPPipeline(
        voxel_size=1.0,
        source_voxel_size=0.3,
        min_points_per_voxel=3,
        alpha=0.1,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        min_motion_th=0.1,
        max_map_points=100_000,
        max_map_frames=500,
        use_fim_weight=use_fim_weight,
        fim_auto_gate=fim_auto_gate,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        max_source_points=0,
        device="cpu",
    )
    poses = []
    times = []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        r = p.process_frame(f[:, :3], f[:, 3], timestamp=float(i))
        times.append((time.perf_counter() - t0) * 1000)
        poses.append(r.pose.copy())
        if i % 200 == 0:
            print(f"  {label}  {i}/{len(frames)}  {np.mean(times[-20:]):.0f}ms", end="\r")
    print(f"  {label}  done  mean={np.mean(times):.1f}ms        ")
    return poses, times


def compute_ate(est_poses, gt_poses):
    """Umeyama-aligned translation RMSE."""
    P = np.array([T[:3, 3] for T in est_poses]).T
    Q = np.array([T[:3, 3] for T in gt_poses[:len(est_poses)]]).T
    mu_p = P.mean(1, keepdims=True); mu_q = Q.mean(1, keepdims=True)
    H = (P - mu_p) @ (Q - mu_q).T
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_q - R @ mu_p
    return float(np.sqrt(np.mean(np.linalg.norm(R @ P + t - Q, axis=0) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="00")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--configs", nargs="+",
                    default=["c1_off", "c1_on", "g005", "g020", "g050"])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from run_kitti_benchmark import load_kitti_sequence

    frames, gt_poses = load_kitti_sequence(args.seq, args.max_frames)
    print(f"[KITTI seq{args.seq}] {len(frames)} frames, {len(gt_poses)} GT poses")

    config_map = {
        "c1_off": (False, 0.0),
        "c1_on":  (True,  0.0),
        "g005":   (True,  0.005),
        "g020":   (True,  0.020),
        "g050":   (True,  0.050),
    }

    out = OUT_ROOT / f"seq{args.seq}"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for cfg_name in args.configs:
        fim_w, gate = config_map[cfg_name]
        label = f"IV-{cfg_name}"
        poses, times = run_iv(frames, fim_w, gate, label)
        ate = compute_ate(poses, gt_poses)
        results[cfg_name] = {
            "ate": ate,
            "ms_mean": float(np.mean(times)),
            "use_fim_weight": fim_w,
            "fim_auto_gate": gate,
            "n_frames": len(frames),
        }
        print(f"  → ATE={ate:.4f}m")

    (out / "autogate.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"KITTI seq{args.seq} ({len(frames)} fr) auto-gate regression")
    print(f"{'Config':<10} {'ATE[m]':>8}  {'ms/fr':>7}")
    for k, r in results.items():
        print(f"  {k:<8} {r['ate']:>8.4f}  {r['ms_mean']:>7.1f}")
    print(f"\nSaved → {out}/autogate.json")


if __name__ == "__main__":
    main()
