#!/usr/bin/env python3
"""
KITTI Odometry Benchmark: IV-GICP vs KISS-ICP (optional GenZ-ICP binary)

Usage:
    uv run python examples/run_kitti_benchmark.py --seq 00 --max-frames 100
    uv run python examples/run_kitti_benchmark.py --seq 00 --sparse-speed
    uv run python examples/run_kitti_benchmark.py --seq 00 --sparse-extreme
    # --sparse-speed: balanced sparse C4.  --sparse-extreme: LOAM-like minimal features (ATE risk ↑).
    # Default --max-frames 100: fast IV vs KISS comparison (raise for long-horizon ATE).
    # Speed budget vs KISS (mean ms/frame): ≤1.5× preferred, >2× should be avoided.
    # IV-GICP defaults to --device cpu (C++ registration). Use --device cuda only for Python/GPU path (much slower).

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
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

KITTI_ROOT   = Path("/home/km/data/kitti/dataset")
GENZ_BINARY  = Path(__file__).parent.parent / "thirdparty/genz-icp/kitti_runner/build/genz_kitti_runner"
RESULTS_ROOT = Path(__file__).parent.parent / "results" / "kitti"

# IV-GICP vs KISS mean ms/frame: ≤1.5× acceptable; >2× is considered too slow for iterative tuning.
SPEED_RATIO_SOFT = 1.5
SPEED_RATIO_HARD = 2.0

# LOAM-style: cap C4 output + tight GN/map budget (speed first). If ATE blows up, relax
# source_min_feature_score / plane thresholds or raise source_max_output_features first.
# Sparse C4 yields few points per 1m map voxel — default min_points_per_voxel=5 can leave
# zero valid target voxels → reg_ms=0. Use 3 for map eligibility (see docs/speed_first_sparse.md).
SPARSE_SPEED_PRESET = dict(
    min_points_per_voxel=3,
    max_registration_target_voxels=8192,
    source_voxel_size=0.55,
    max_map_frames=180,
    auto_alpha=False,
    max_iterations=10,
    max_source_points=1024,
    source_max_output_features=768,
    source_min_feature_score=0.01,
    source_drop_small_voxels=True,
    use_fim_weight=False,
    use_entropy_alpha=False,
    kdtree_interval=8,
    stable_frames=8,
    source_plane_planarity_thresh=0.12,
    source_plane_linearity_thresh=0.12,
)

# Fewer points / stronger P·L gate than SPARSE_SPEED_PRESET. Tuned so seq00 short runs do not
# always diverge; for even sparser LOAM-style, lower source_max_output_features in a local fork.
SPARSE_EXTREME_PRESET = dict(
    min_points_per_voxel=3,
    max_registration_target_voxels=6144,
    source_voxel_size=0.58,
    max_map_frames=160,
    auto_alpha=False,
    max_iterations=9,
    max_source_points=896,
    source_max_output_features=640,
    source_min_feature_score=0.018,
    source_drop_small_voxels=True,
    use_fim_weight=False,
    use_entropy_alpha=False,
    kdtree_interval=8,
    stable_frames=8,
    source_plane_planarity_thresh=0.125,
    source_plane_linearity_thresh=0.125,
)


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

def run_iv_gicp(frames, device, alpha, label, sparse_mode: Optional[str] = None, **kw):
    from iv_gicp.pipeline import IVGICPPipeline
    params = dict(
        voxel_size=1.0,
        source_voxel_size=0.3,
        min_points_per_voxel=3,
        alpha=alpha,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        min_motion_th=0.1,
        max_map_points=100_000,
        max_map_frames=500,
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        max_source_points=0,
        device=device,
    )
    if sparse_mode == "extreme":
        params.update(SPARSE_EXTREME_PRESET)
    elif sparse_mode == "speed":
        params.update(SPARSE_SPEED_PRESET)
    params.update(kw)
    pipeline = IVGICPPipeline(**params)
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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Number of frames (default 100 for quicker IV vs KISS runs).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="IV-GICP only: 'cpu' uses C++ core (fast). 'cuda'/'auto' often trigger the slower Python+GPU registration path.",
    )
    parser.add_argument("--skip-genz",   action="store_true", help="Skip GenZ-ICP (slow build)")
    sp = parser.add_mutually_exclusive_group()
    sp.add_argument(
        "--sparse-speed",
        action="store_true",
        help="IV-GICP: sparse C4 (768 cap, mild score gate) + tight GN/map — speed-first, safer ATE.",
    )
    sp.add_argument(
        "--sparse-extreme",
        action="store_true",
        help="IV-GICP: LOAM-like minimal C4 (512 cap, stricter P/L) — fastest sparse; ATE may degrade.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device  # prefer "cpu" for IV-GICP C++ path

    out = RESULTS_ROOT / f"seq{args.seq}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  KITTI Odometry Benchmark — seq {args.seq}")
    print(f"  data: {KITTI_ROOT}")
    iv_sparse = "extreme" if args.sparse_extreme else ("speed" if args.sparse_speed else None)
    print(f"  device: {device}  max_frames: {args.max_frames}  iv_sparse_mode: {iv_sparse!r}")
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

    # ── IV-GICP (alpha=0.1) ───────────────────────────────────────────────────
    # α=0.5 → omega_I≈225 >> Omega_geo_xx≈1 → intensity dominates geometry → bad.
    # α=0.1 → omega_I≈9 ≈ Omega_geo_xx → balanced geo+intensity contribution.
    if iv_sparse == "extreme":
        iv_lab = "IV-GICP (sparse-extreme)"
        iv_key = "IV-GICP-sparse-extreme"
    elif iv_sparse == "speed":
        iv_lab = "IV-GICP (sparse-speed)"
        iv_key = "IV-GICP-sparse"
    else:
        iv_lab = "IV-GICP"
        iv_key = "IV-GICP"
    iv_poses, iv_times, iv_reg, iv_map = run_iv_gicp(
        frames, device, alpha=0.1, label=iv_lab, sparse_mode=iv_sparse,
    )
    iv_ate, iv_ate_m = ate_rmse(iv_poses, poses_gt) if poses_gt else (float("nan"), float("nan"))
    iv_kt = kitti_t_err(iv_poses, poses_gt) if poses_gt else float("nan")
    results[iv_key] = dict(
        ate_rmse=iv_ate, ate_mean=iv_ate_m, kitti_t_err=iv_kt,
        mean_ms=float(np.mean(iv_times[1:])), n_frames=n, device=device,
        reg_ms=float(np.mean(iv_reg[1:])), map_ms=float(np.mean(iv_map[1:])),
        sparse_mode=iv_sparse,
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

    # Speed vs KISS (skip frame 0 for both — cold start)
    kiss_ms = float(np.mean(ki_times[1:])) if n > 1 else float(np.mean(ki_times))
    iv_ms = float(np.mean(iv_times[1:])) if n > 1 else float(np.mean(iv_times))
    speed_ratio = iv_ms / kiss_ms if kiss_ms > 1e-9 else float("nan")
    speed_ok_soft = speed_ratio <= SPEED_RATIO_SOFT
    speed_ok_hard = speed_ratio <= SPEED_RATIO_HARD
    print(f"\n  Speed vs KISS (mean ms/fr, frame≥1): IV/KISS = {speed_ratio:.2f}×  "
          f"(목표 ≤{SPEED_RATIO_SOFT:.1f}×, 상한 {SPEED_RATIO_HARD:.1f}×)")
    if not speed_ok_hard:
        print(f"  [경고] IV-GICP가 KISS 대비 {SPEED_RATIO_HARD:.1f}×를 초과했습니다. 속도 개선이 필요합니다.")
    elif not speed_ok_soft:
        print(f"  [참고] {SPEED_RATIO_SOFT:.1f}× 초과 — 허용 가능하지만 여유 있으면 최적화 권장.")

    payload = {
        "seq": args.seq,
        "n_frames": n,
        "iv_sparse_mode": iv_sparse,
        "methods": results,
        "speed_vs_kiss": {
            "kiss_mean_ms": kiss_ms,
            "iv_mean_ms": iv_ms,
            "ratio_iv_over_kiss": speed_ratio,
            "target_ratio_max": SPEED_RATIO_SOFT,
            "hard_cap_ratio": SPEED_RATIO_HARD,
            "within_soft_budget": speed_ok_soft,
            "within_hard_cap": speed_ok_hard,
        },
    }
    with open(out / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {out}/results.json")


if __name__ == "__main__":
    main()
