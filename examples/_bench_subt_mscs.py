"""SubT-MRS benchmark: IV-GICP default vs MSCS vs KISS-ICP.

Focuses on Final_UGV1 (mine/tunnel) — degenerate underground geometry.

Usage:
    uv run python examples/_bench_subt_mscs.py              # Final_UGV1 100fr
    uv run python examples/_bench_subt_mscs.py --frames 200
"""
import sys, time, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_subt_eval import (
    DATASETS, BASE_DIR, load_gt_csv, load_frames_from_zipped_bags, compute_ate
)

DATASET_NAME = "Final_UGV1"
MAX_FRAMES = 100

# SubT Final_UGV1: mine/tunnel, VLP-16 (run_full_eval.py _subt_iv() 기준)
BASE = dict(
    voxel_size=0.5,
    source_voxel_size=0.3,
    alpha=0.1,
    max_correspondence_distance=2.0,
    initial_threshold=1.5,
    max_map_frames=200,
    map_radius=200.0,
    min_motion_th=0.5,
    max_iterations=30,
    min_range=0.3,
    max_range=80.0,
    auto_alpha=False,
    device='cpu',
)


def ate_rmse(pred_poses, gt_poses):
    """ATE RMSE after Umeyama alignment."""
    n = min(len(pred_poses), len(gt_poses))
    if n < 2:
        return float("nan")
    t_est = np.array([p[:3, 3] for p in pred_poses[:n]])
    t_gt  = np.array([p[:3, 3] for p in gt_poses[:n]])
    mu_e = t_est.mean(0); mu_g = t_gt.mean(0)
    H = (t_est - mu_e).T @ (t_gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ D @ U.T
    t = mu_g - R @ mu_e
    errs = np.array([np.linalg.norm(R @ p[:3, 3] + t - gt[:3, 3])
                     for p, gt in zip(pred_poses[:n], gt_poses[:n])])
    return float(np.sqrt(np.mean(errs**2)))


def run_ivgicp(frames, gt_ts, use_mscs=False, mscs_kappa_max=100.0):
    from iv_gicp.pipeline import IVGICPPipeline
    cfg = dict(**BASE, use_mscs=use_mscs, mscs_kappa_max=mscs_kappa_max)
    pipeline = IVGICPPipeline(**cfg)
    poses, times, kappas, mscs_ratios, est_ts = [], [], [], [], []
    for ts_ns, pts in frames:
        xyz = pts[:, :3]; ints = pts[:, 3]
        t0 = time.perf_counter()
        result = pipeline.process_frame(xyz, ints)
        elapsed = (time.perf_counter() - t0) * 1000
        poses.append(result.pose.copy())
        times.append(elapsed)
        kappas.append(result.kappa)
        mscs_ratios.append(result.mscs_ratio)
        est_ts.append(ts_ns)

    # ATE via timestamp-matched GT
    ate = float("nan")
    if gt_ts is not None and len(gt_ts[0]) > 0:
        gt_timestamps_ns, gt_poses = gt_ts
        ate, n_matched = compute_ate(poses, gt_timestamps_ns, gt_poses, est_ts)

    return poses, times, kappas, mscs_ratios, ate


def run_kiss(frames, gt_ts):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.mapping.voxel_size = BASE['voxel_size']
    cfg.data.deskew = False
    cfg.data.max_range = 50.0
    cfg.data.min_range = 0.5
    od = KissICP(config=cfg)
    poses, times, est_ts = [], [], []
    for i, (ts_ns, pts) in enumerate(frames):
        xyz = pts[:, :3]
        t0 = time.perf_counter()
        od.register_frame(xyz, np.zeros(len(xyz)))
        elapsed = (time.perf_counter() - t0) * 1000
        poses.append(od.last_pose.copy())
        times.append(elapsed)
        est_ts.append(ts_ns)

    ate = float("nan")
    if gt_ts is not None and len(gt_ts[0]) > 0:
        gt_timestamps_ns, gt_poses = gt_ts
        ate, _ = compute_ate(poses, gt_timestamps_ns, gt_poses, est_ts)

    return poses, times, ate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    info = DATASETS[DATASET_NAME]
    rosbag_zip = Path(BASE_DIR) / info["rosbag"]
    gt_zip     = Path(BASE_DIR) / info["gt"]

    if not rosbag_zip.exists():
        print(f"Rosbag not found: {rosbag_zip}")
        return

    print(f"Dataset : {DATASET_NAME} ({info['env']})")
    print(f"Rosbag  : {rosbag_zip}")
    print(f"GT zip  : {gt_zip}")
    print(f"Frames  : {args.frames}")

    # Load GT
    gt_ts = None
    if gt_zip.exists():
        try:
            gt_timestamps_ns, gt_poses = load_gt_csv(str(gt_zip), info["gt_file"])
            gt_ts = (gt_timestamps_ns, gt_poses)
            print(f"GT loaded: {len(gt_timestamps_ns)} poses")
        except Exception as e:
            print(f"GT load failed: {e}")

    # Load frames once (generator → list)
    print("\nLoading frames...", flush=True)
    frames = list(load_frames_from_zipped_bags(str(rosbag_zip), max_frames=args.frames))
    print(f"Loaded {len(frames)} frames")

    print(f"\n{'='*65}")
    print(f"  SubT {DATASET_NAME} — {len(frames)}fr")
    print(f"{'='*65}")

    print("  [1/3] IV-GICP default ...", flush=True)
    p1, t1, k1, r1, ate1 = run_ivgicp(frames, gt_ts, use_mscs=False)
    ms1 = np.mean(t1[1:])

    print("  [2/3] IV-GICP + MSCS  ...", flush=True)
    p2, t2, k2, r2, ate2 = run_ivgicp(frames, gt_ts, use_mscs=True, mscs_kappa_max=100.0)
    ms2 = np.mean(t2[1:])

    print("  [3/3] KISS-ICP        ...", flush=True)
    p3, t3, ate3 = run_kiss(frames, gt_ts)
    ms3 = np.mean(t3[1:])

    print(f"\n  {'Method':<28} {'ms/fr':>7} {'Hz':>6} {'ATE':>9} {'κ':>8} {'MSCS%':>7}")
    print(f"  {'-'*63}")

    for label, poses, times, kappas, mscs_ratios, ate in [
        ("IV-GICP default",  p1, t1, k1, r1, ate1),
        ("IV-GICP + MSCS",   p2, t2, k2, r2, ate2),
    ]:
        ms   = np.mean(times[1:])
        km   = np.mean(kappas[1:]) if kappas else 0
        mpct = np.mean(mscs_ratios[1:]) * 100 if mscs_ratios else 100
        ate_str = f"{ate:.3f}m" if not np.isnan(ate) else "N/A"
        diff_str = ""
        if not np.isnan(ate) and not np.isnan(ate3) and ate3 > 0:
            diff_str = f"({(ate/ate3-1)*100:+.1f}%)"
        print(f"  {label:<28} {ms:>7.1f} {1000/ms:>6.1f} {ate_str:>7} {diff_str:<9} {km:>8.0f} {mpct:>6.1f}%")

    ms3_avg = np.mean(t3[1:])
    ate3_str = f"{ate3:.3f}m" if not np.isnan(ate3) else "N/A"
    print(f"  {'KISS-ICP':<28} {ms3_avg:>7.1f} {1000/ms3_avg:>6.1f} {ate3_str:>7} {'(ref)':<9} {'N/A':>8} {'N/A':>7}")

    pcts = np.array(r2[1:]) * 100
    print(f"\n  MSCS ratio: mean={pcts.mean():.1f}%  p10={np.percentile(pcts,10):.1f}%  p90={np.percentile(pcts,90):.1f}%")
    if k1:
        kk = np.array(k1[1:])
        print(f"  κ (default): mean={kk.mean():.0f}  p10={np.percentile(kk,10):.0f}  p90={np.percentile(kk,90):.0f}")

    out = dict(
        dataset=DATASET_NAME, max_frames=args.frames,
        ivgicp=dict(ms=float(ms1), ate=ate1, kappa=float(np.mean(k1[1:])) if k1 else None),
        ivgicp_mscs=dict(ms=float(ms2), ate=ate2, mscs_pct=float(pcts.mean())),
        kiss=dict(ms=float(ms3), ate=ate3),
    )
    Path("results").mkdir(exist_ok=True)
    with open("results/bench_subt_mscs.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/bench_subt_mscs.json")


if __name__ == "__main__":
    main()
