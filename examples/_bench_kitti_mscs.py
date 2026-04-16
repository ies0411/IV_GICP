"""KITTI odometry benchmark: IV-GICP default vs MSCS vs KISS-ICP.

Usage:
    uv run python examples/_bench_kitti_mscs.py              # seq00 + seq05
    uv run python examples/_bench_kitti_mscs.py --seqs 00    # seq00 only
    uv run python examples/_bench_kitti_mscs.py --frames 200
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

KITTI_ROOT = Path("/home/km/deepai_dev_data/kitti/dataset")
DEFAULT_SEQS = ["00", "05"]
MAX_FRAMES = 100


def read_kitti_frames(seq, max_frames=None):
    bin_dir = KITTI_ROOT / "sequences" / seq / "velodyne"
    files = sorted(bin_dir.glob("*.bin"))[:max_frames]
    frames = []
    for f in files:
        raw = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        r = np.linalg.norm(raw[:, :3], axis=1)
        frames.append(raw[(r > 0.5) & (r < 80.0)].astype(np.float64))
    print(f"  Loaded {len(frames)} frames from KITTI seq{seq}")
    return frames


def load_kitti_gt(seq):
    gt_path = KITTI_ROOT / "poses" / f"{seq}.txt"
    poses = []
    with open(gt_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def ate_rmse(pred_poses, gt_poses):
    """ATE RMSE after Umeyama alignment (handles LiDAR/camera frame difference)."""
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


def run_ivgicp(frames, use_mscs=False, mscs_kappa_max=100.0):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=1.0,
        source_voxel_size=0.3,
        alpha=0.1,
        max_correspondence_distance=2.0,
        initial_threshold=2.0,
        max_map_frames=500,
        max_iterations=30,
        min_range=0.5,
        max_range=80.0,
        auto_alpha=False,
        min_motion_th=0.1,
        device='cpu',
        use_mscs=use_mscs,
        mscs_kappa_max=mscs_kappa_max,
    )
    poses, times, kappas, mscs_ratios = [], [], [], []
    for raw in frames:
        xyz = raw[:, :3]; ints = raw[:, 3]
        t0 = time.perf_counter()
        result = pipeline.process_frame(xyz, ints)
        elapsed = (time.perf_counter() - t0) * 1000
        poses.append(result.pose.copy())
        times.append(elapsed)
        kappas.append(result.kappa)
        mscs_ratios.append(result.mscs_ratio)
    return poses, times, kappas, mscs_ratios


def run_kiss(frames):
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.mapping.voxel_size = 1.0
    cfg.data.deskew = False
    cfg.data.max_range = 80.0
    cfg.data.min_range = 0.5
    od = KissICP(config=cfg)
    poses, times = [], []
    for raw in frames:
        t0 = time.perf_counter()
        od.register_frame(raw[:, :3], np.zeros(len(raw)))
        elapsed = (time.perf_counter() - t0) * 1000
        poses.append(od.last_pose.copy())
        times.append(elapsed)
    return poses, times


def bench_seq(seq, max_frames):
    print(f"\n{'='*65}")
    print(f"  KITTI seq{seq} — {max_frames}fr")
    print(f"{'='*65}")

    frames = read_kitti_frames(seq, max_frames)
    gt = load_kitti_gt(seq)[:max_frames]

    print("  [1/3] IV-GICP default ...", flush=True)
    p1, t1, k1, r1 = run_ivgicp(frames, use_mscs=False)
    ms1, ate1 = np.mean(t1[1:]), ate_rmse(p1, gt)

    print("  [2/3] IV-GICP + MSCS  ...", flush=True)
    p2, t2, k2, r2 = run_ivgicp(frames, use_mscs=True)
    ms2, ate2 = np.mean(t2[1:]), ate_rmse(p2, gt)

    print("  [3/3] KISS-ICP        ...", flush=True)
    p3, t3 = run_kiss(frames)
    ms3, ate3 = np.mean(t3[1:]), ate_rmse(p3, gt)

    print(f"\n  {'Method':<28} {'ms/fr':>7} {'Hz':>6} {'ATE':>9} {'κ':>8} {'MSCS%':>7}")
    print(f"  {'-'*63}")
    for label, poses, times, kappas, mscs_ratios, ate in [
        ("IV-GICP default",  p1, t1, k1, r1, ate1),
        ("IV-GICP + MSCS",   p2, t2, k2, r2, ate2),
    ]:
        ms   = np.mean(times[1:])
        km   = np.mean(kappas[1:])
        mpct = np.mean(mscs_ratios[1:]) * 100
        diff = f"({(ate/ate3-1)*100:+.1f}%)" if ate3 > 0 else ""
        print(f"  {label:<28} {ms:>7.1f} {1000/ms:>6.1f} {ate:>7.3f}m {diff:<9} {km:>8.0f} {mpct:>6.1f}%")
    ms = np.mean(t3[1:])
    print(f"  {'KISS-ICP':<28} {ms:>7.1f} {1000/ms:>6.1f} {ate3:>7.3f}m {'(ref)':<9} {'N/A':>8} {'N/A':>7}")

    pcts = np.array(r2[1:]) * 100
    print(f"\n  MSCS ratio: mean={pcts.mean():.1f}%  p10={np.percentile(pcts,10):.1f}%  p90={np.percentile(pcts,90):.1f}%")

    return dict(
        seq=seq, max_frames=max_frames,
        ivgicp=dict(ms=float(ms1), ate=ate1, kappa=float(np.mean(k1[1:]))),
        ivgicp_mscs=dict(ms=float(ms2), ate=ate2, mscs_pct=float(pcts.mean())),
        kiss=dict(ms=float(ms3), ate=ate3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=DEFAULT_SEQS)
    ap.add_argument("--frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    all_results = {}
    for seq in args.seqs:
        all_results[seq] = bench_seq(seq, args.frames)

    Path("results").mkdir(exist_ok=True)
    with open("results/bench_kitti_mscs.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/bench_kitti_mscs.json")


if __name__ == "__main__":
    main()
