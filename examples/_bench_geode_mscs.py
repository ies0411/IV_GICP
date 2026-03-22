"""Quick GEODE Urban_Tunnel01 benchmark: IV-GICP default vs MSCS vs KISS-ICP (100fr)."""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_geode_eval import read_geode_frames, load_geode_gt, interpolate_gt_at_lidar_times

GEODE_ROOT = Path("/home/km/data/GEODE")
SEQ = "Urban_Tunnel01"
MAX_FRAMES = 100

# Base config for GEODE Urban Tunnel (run_full_eval.py _geode_urban_iv() 기준)
BASE = dict(
    voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
    max_correspondence_distance=2.0, initial_threshold=1.5,
    map_radius=80.0, min_motion_th=0.5,
    max_map_frames=500, max_iterations=30,
    min_range=0.5, max_range=80.0,
    auto_alpha=False,
    device='cpu',
)

def ate_rmse(pred_poses, gt_poses):
    """ATE RMSE after Umeyama alignment."""
    n = min(len(pred_poses), len(gt_poses))
    if n < 2:
        return float("nan"), float("nan")
    t_est = np.array([p[:3, 3] for p in pred_poses[:n]])
    t_gt  = np.array([p[:3, 3] for p in gt_poses[:n]])
    mu_e = t_est.mean(0); mu_g = t_gt.mean(0)
    H = (t_est - mu_e).T @ (t_gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ D @ U.T
    t = mu_g - R @ mu_e
    errs = np.array([np.linalg.norm(R @ p[:3,3] + t - gt[:3,3])
                     for p, gt in zip(pred_poses[:n], gt_poses[:n])])
    return float(np.sqrt(np.mean(errs**2))), float(np.mean(errs))


def run_ivgicp(frames, use_mscs=False, mscs_kappa_max=100.0, max_source_points=2048):
    from iv_gicp.pipeline import IVGICPPipeline
    cfg = dict(**BASE, use_mscs=use_mscs, mscs_kappa_max=mscs_kappa_max,
               max_source_points=max_source_points)
    pipeline = IVGICPPipeline(**cfg)
    poses, times, kappas, mscs_ratios = [], [], [], []
    for ts, pts in frames:
        xyz = pts[:, :3]; ints = pts[:, 3]
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
    cfg.mapping.voxel_size = BASE['voxel_size']
    cfg.data.deskew = False
    cfg.data.max_range = 80.0
    cfg.data.min_range = 0.5
    od = KissICP(config=cfg)
    poses, times = [], []
    for i, (ts, pts) in enumerate(frames):
        xyz = pts[:, :3]
        t0 = time.perf_counter()
        od.register_frame(xyz, np.full(len(xyz), float(i)))
        elapsed = (time.perf_counter() - t0) * 1000
        poses.append(od.last_pose.copy())
        times.append(elapsed)
    return poses, times


def main():
    # GEODE data layout:
    #   sensor_data/Urban_tunnel/Urban_TunnelXX/Urban_TunnelXX.bag
    #   groundtruth/Urban_tunnel/Urban_TunnelXX.txt
    bag_path = GEODE_ROOT / "sensor_data" / "Urban_tunnel" / SEQ / f"{SEQ}.bag"
    gt_path  = GEODE_ROOT / "groundtruth"  / "Urban_tunnel" / f"{SEQ}.txt"
    if not bag_path.exists():
        print(f"Bag not found: {bag_path}")
        return

    print(f"Bag: {bag_path}")
    print(f"GT:  {gt_path}")

    frames = read_geode_frames(bag_path, max_frames=MAX_FRAMES)
    lidar_times = np.array([f[0] for f in frames])

    gt_aligned = None
    if gt_path.exists():
        try:
            gt_times, gt_poses_local = load_geode_gt(gt_path)
            gt_aligned = interpolate_gt_at_lidar_times(gt_times, gt_poses_local, lidar_times)
            print(f"GT loaded: {len(gt_times)} poses")
        except Exception as e:
            print(f"GT load failed: {e}")

    print(f"\n{'='*60}")
    print(f"GEODE {SEQ} — {MAX_FRAMES}fr benchmark")
    print(f"{'='*60}")

    # Run all three in sequence (can't easily parallelize in single process)
    print("\n[1/3] IV-GICP default (use_mscs=False)...")
    p1, t1, k1, r1 = run_ivgicp(frames, use_mscs=False)
    ms1 = np.mean(t1[1:])
    print(f"  {ms1:.1f} ms/frame  ({1000/ms1:.1f} Hz)")

    print("\n[2/3] IV-GICP + MSCS (use_mscs=True, kappa_max=100)...")
    p2, t2, k2, r2 = run_ivgicp(frames, use_mscs=True, mscs_kappa_max=100.0, max_source_points=4096)
    ms2 = np.mean(t2[1:])
    print(f"  {ms2:.1f} ms/frame  ({1000/ms2:.1f} Hz)")

    print("\n[3/3] KISS-ICP...")
    p3, t3 = run_kiss(frames)
    ms3 = np.mean(t3[1:])
    print(f"  {ms3:.1f} ms/frame  ({1000/ms3:.1f} Hz)")

    print(f"\n{'='*60}")
    print(f"{'Method':<30} {'ms/fr':>7} {'Hz':>6} {'ATE':>10} {'κ_mean':>10} {'MSCS%':>8}")
    print(f"{'-'*60}")

    rows = []
    for label, poses, times, kappas, mscs_ratios in [
        ("IV-GICP default", p1, t1, k1, r1),
        ("IV-GICP + MSCS", p2, t2, k2, r2),
    ]:
        ms = np.mean(times[1:])
        hz = 1000 / ms
        ate_str = "N/A"
        if gt_aligned is not None:
            ate_val, _ = ate_rmse(poses, gt_aligned)
            ate_str = f"{ate_val:.3f}m"
        k_mean = np.mean(kappas[1:]) if kappas else 0
        mscs_pct = np.mean(mscs_ratios[1:]) * 100 if mscs_ratios else 100
        print(f"  {label:<28} {ms:>7.1f} {hz:>6.1f} {ate_str:>10} {k_mean:>10.1f} {mscs_pct:>7.1f}%")
        rows.append(dict(label=label, ms=ms, hz=hz, ate=ate_str, kappa_mean=k_mean, mscs_pct=mscs_pct))

    ms = np.mean(t3[1:])
    hz = 1000 / ms
    ate_str = "N/A"
    if gt_aligned is not None:
        ate_val, _ = ate_rmse(p3, gt_aligned)
        ate_str = f"{ate_val:.3f}m"
    print(f"  {'KISS-ICP':<28} {ms:>7.1f} {hz:>6.1f} {ate_str:>10} {'N/A':>10} {'N/A':>8}")
    rows.append(dict(label="KISS-ICP", ms=ms, hz=hz, ate=ate_str, kappa_mean=None, mscs_pct=None))

    print(f"\nMSCS ratio distribution (IV-GICP+MSCS, frames 1+):")
    if r2:
        pcts = np.array(r2[1:]) * 100
        print(f"  mean={pcts.mean():.1f}%  median={np.median(pcts):.1f}%  "
              f"p10={np.percentile(pcts,10):.1f}%  p90={np.percentile(pcts,90):.1f}%")

    print(f"\nκ distribution (IV-GICP default, frames 1+):")
    if k1:
        kk = np.array(k1[1:])
        print(f"  mean={kk.mean():.1f}  median={np.median(kk):.1f}  "
              f"p10={np.percentile(kk,10):.1f}  p90={np.percentile(kk,90):.1f}")

    out = dict(seq=SEQ, max_frames=MAX_FRAMES, rows=rows)
    Path("results").mkdir(exist_ok=True)
    with open("results/bench_geode_mscs.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/bench_geode_mscs.json")


if __name__ == "__main__":
    main()
