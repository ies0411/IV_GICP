"""
Metro Tunnel Parameter Grid Search
===================================
Quick grid search on seq1, 200 frames to find better IV-GICP config for Livox Mid-360.

Usage:
    uv run python examples/_metro_grid_search.py --max-frames 200 --device cpu
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

GEODE_ROOT = Path("/home/km/data/GEODE")
LIDAR_TOPIC = "/livox/lidar"


# ── Livox frame parser (copied from run_metro_eval.py) ────────────────────────

def parse_livox_frame(raw: bytes, max_range: float = 60.0):
    data = bytes(raw)
    secs, nsecs = struct.unpack_from('<II', data, 4)
    fid_len = struct.unpack_from('<I', data, 12)[0]
    off = 16 + fid_len
    off += 8 + 4 + 1 + 3
    arr_len = struct.unpack_from('<I', data, off)[0]
    off += 4
    arr = np.frombuffer(data[off:off + 19 * arr_len], dtype=np.uint8).reshape(arr_len, 19)
    x    = arr[:,  4: 8].view(np.float32).reshape(-1)
    y    = arr[:,  8:12].view(np.float32).reshape(-1)
    z    = arr[:, 12:16].view(np.float32).reshape(-1)
    refl = arr[:, 16].astype(np.float64) / 255.0
    pts = np.stack([x, y, z, refl], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    ts_ns = secs * 1e9 + nsecs
    return ts_ns, pts[valid]


def read_metro_frames(bag_path: Path, max_frames: int = None, max_range: float = 60.0):
    from rosbags.rosbag1 import Reader
    frames = []
    print(f"Reading: {bag_path}")
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = conns[0].msgcount if conns else 0
        print(f"  Livox frames in bag: {total}")
        for conn, ts_ns, raw in bag.messages(connections=conns):
            try:
                t_ns, pts = parse_livox_frame(raw, max_range=max_range)
                if pts is not None and len(pts) > 100:
                    frames.append((t_ns / 1e9, pts))
                    if max_frames and len(frames) >= max_frames:
                        break
            except Exception:
                continue
    print(f"  Done: {len(frames)} frames  avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame")
    return frames


def load_gt(gt_path: Path):
    from scipy.spatial.transform import Rotation
    data = np.loadtxt(gt_path)
    timestamps = data[:, 0]
    txyz  = data[:, 1:4]
    quats = data[:, 4:8]
    T0_inv = None
    poses  = []
    for i in range(len(timestamps)):
        q = quats[i]
        if np.linalg.norm(q) < 1e-6:
            R = np.eye(3)
        else:
            R = Rotation.from_quat(q).as_matrix()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = txyz[i]
        if T0_inv is None:
            T0_inv = np.linalg.inv(T)
        poses.append(T0_inv @ T)
    return timestamps, np.stack(poses)


def interpolate_gt(gt_times, gt_poses, lidar_times):
    idx  = np.searchsorted(gt_times, lidar_times)
    idx  = np.clip(idx, 0, len(gt_times) - 1)
    prev = np.clip(idx - 1, 0, len(gt_times) - 1)
    dt_b = np.abs(gt_times[prev] - lidar_times)
    dt_a = np.abs(gt_times[idx]  - lidar_times)
    best = np.where(dt_b < dt_a, prev, idx)
    return gt_poses[best]


def save_tum(poses, timestamps, path: Path):
    from scipy.spatial.transform import Rotation
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def traj_stats(poses):
    dists = [np.linalg.norm(poses[i+1][:3, 3] - poses[i][:3, 3])
             for i in range(len(poses) - 1)]
    return float(np.sum(dists)), float(np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]))


def compose_poses(rel_list):
    poses = [np.eye(4)]
    for T in rel_list:
        poses.append(poses[-1] @ T)
    return poses


def compute_ate(gt_tum: Path, pred_tum: Path):
    try:
        from evo.tools import file_interface
        from evo.core import sync, metrics
        ref = file_interface.read_tum_trajectory_file(str(gt_tum))
        est = file_interface.read_tum_trajectory_file(str(pred_tum))
        ref, est = sync.associate_trajectories(ref, est)
        est.align(ref, correct_scale=False)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, est))
        return float(ape.get_statistic(metrics.StatisticsType.rmse))
    except Exception as e:
        print(f"    evo ATE failed: {e}")
        return None


def run_config(frames, timestamps, config, device="cpu", tmp_dir=None):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=config["voxel_size"],
        source_voxel_size=config["source_voxel_size"],
        alpha=config["alpha"],
        max_correspondence_distance=config["max_correspondence_distance"],
        initial_threshold=config.get("initial_threshold", 1.5),
        min_motion_th=config.get("min_motion_th", 0.5),
        max_map_frames=config.get("max_map_frames", 200),
        max_iterations=config.get("max_iterations", 20),
        map_radius=config.get("map_radius", 60.0),
        auto_alpha=False,
        auto_alpha_from_intensity=False,
        source_drop_small_voxels=False,
        source_max_output_features=0,
        source_min_feature_score=0.0,
        max_source_points=0,
        device=device,
    )
    abs_poses = [np.eye(4)]
    times = []
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        elapsed = (time.perf_counter() - t0) * 1000
        abs_poses.append(result.pose.copy())
        times.append(elapsed)

    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    iv_poses = compose_poses(rel)

    # Save and compute ATE
    tum_path = tmp_dir / f"pred_{config['tag']}.tum"
    save_tum(iv_poses, timestamps, tum_path)
    path_len, end_disp = traj_stats(iv_poses)
    mean_ms = float(np.mean(times[1:])) if len(times) > 1 else float(np.mean(times))
    return {
        "path_len": path_len,
        "end_disp": end_disp,
        "mean_ms": mean_ms,
        "tum_path": tum_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",        default="1", choices=["1", "2", "3"])
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    seq_bag = f"Shield_tunnel{args.seq}_gamma"
    seq_gt  = f"Shield_tunnel{args.seq}"
    bag_path = GEODE_ROOT / "sensor_data" / "Metro_tunnel" / seq_bag / f"{seq_bag}.bag"
    gt_path  = GEODE_ROOT / "groundtruth" / "metro_tunnel" / f"{seq_gt}.txt"
    tmp_dir  = Path(__file__).parent.parent / "results" / "geode" / "_gridsearch"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    frames = read_metro_frames(bag_path, max_frames=args.max_frames)
    n = len(frames)
    timestamps = [f[0] for f in frames]

    # ── GT ────────────────────────────────────────────────────────────────────
    print(f"\nLoading GT: {gt_path}")
    gt_times, gt_poses = load_gt(gt_path)
    gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
    gt_tum = tmp_dir / "gt.tum"
    save_tum(gt_at_lidar, timestamps, gt_tum)
    gt_path_len, gt_end_disp = traj_stats(gt_at_lidar)
    print(f"  GT: path={gt_path_len:.1f}m  end_disp={gt_end_disp:.1f}m  (seq{args.seq}, {n}fr)")

    # ── Baseline: KISS-ICP ────────────────────────────────────────────────────
    print("\n[Baseline] KISS-ICP")
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig()
    cfg.data.deskew = False
    cfg.data.max_range = 60.0
    cfg.data.min_range = 0.5
    cfg.mapping.voxel_size = 0.8
    od = KissICP(config=cfg)
    kiss_poses = [np.eye(4)]
    kiss_times = []
    for i, (ts, pts) in enumerate(frames):
        t0 = time.perf_counter()
        od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
        kiss_times.append((time.perf_counter() - t0) * 1000)
        kiss_poses.append(od.last_pose.copy())
    kiss_tum = tmp_dir / "kiss.tum"
    save_tum(kiss_poses, timestamps, kiss_tum)
    kiss_path_len, kiss_end_disp = traj_stats(kiss_poses)
    kiss_ate = compute_ate(gt_tum, kiss_tum)
    print(f"  path={kiss_path_len:.1f}m  end={kiss_end_disp:.2f}m  ATE={kiss_ate:.3f}m  "
          f"mean={np.mean(kiss_times):.1f}ms")

    # ── Grid configs ──────────────────────────────────────────────────────────
    # Baseline config (current best)
    base = dict(voxel_size=0.5, source_voxel_size=0.2, alpha=0.5,
                max_correspondence_distance=1.5, initial_threshold=1.5,
                min_motion_th=0.5, max_map_frames=200, max_iterations=20,
                map_radius=60.0, tag="base")

    grid = [
        # 0. Baseline
        {**base, "tag": "0_baseline"},

        # 1. alpha scan
        {**base, "alpha": 0.0,  "tag": "1a_alpha0.0"},
        {**base, "alpha": 0.1,  "tag": "1b_alpha0.1"},
        {**base, "alpha": 0.2,  "tag": "1c_alpha0.2"},
        {**base, "alpha": 1.0,  "tag": "1d_alpha1.0"},
        {**base, "alpha": 2.0,  "tag": "1e_alpha2.0"},

        # 2. map_radius scan
        {**base, "map_radius": None,  "tag": "2a_mr_None"},
        {**base, "map_radius": 40.0,  "tag": "2b_mr40"},
        {**base, "map_radius": 80.0,  "tag": "2c_mr80"},
        {**base, "map_radius": 100.0, "tag": "2d_mr100"},
        {**base, "map_radius": 150.0, "tag": "2e_mr150"},

        # 3. source_voxel_size scan
        {**base, "source_voxel_size": 0.1, "tag": "3a_sv0.1"},
        {**base, "source_voxel_size": 0.3, "tag": "3b_sv0.3"},
        {**base, "source_voxel_size": 0.4, "tag": "3c_sv0.4"},

        # 4. max_correspondence_distance scan
        {**base, "max_correspondence_distance": 1.0, "tag": "4a_mc1.0"},
        {**base, "max_correspondence_distance": 2.0, "tag": "4b_mc2.0"},
        {**base, "max_correspondence_distance": 3.0, "tag": "4c_mc3.0"},

        # 5. voxel_size scan
        {**base, "voxel_size": 0.3, "source_voxel_size": 0.15, "tag": "5a_vox0.3"},
        {**base, "voxel_size": 0.7, "source_voxel_size": 0.25, "tag": "5b_vox0.7"},
        {**base, "voxel_size": 1.0, "source_voxel_size": 0.3,  "tag": "5c_vox1.0"},

        # 6. max_map_frames scan
        {**base, "max_map_frames": 50,  "tag": "6a_mf50"},
        {**base, "max_map_frames": 100, "tag": "6b_mf100"},
        {**base, "max_map_frames": 500, "tag": "6c_mf500"},

        # 7. min_motion_th scan
        {**base, "min_motion_th": 0.1, "tag": "7a_mth0.1"},
        {**base, "min_motion_th": 0.3, "tag": "7b_mth0.3"},
        {**base, "min_motion_th": 1.0, "tag": "7c_mth1.0"},

        # 8. Combination: alpha=0.0 + mc=2.0 (geometry-only, wider search)
        {**base, "alpha": 0.0, "max_correspondence_distance": 2.0,
         "tag": "8a_alpha0_mc2"},
        # 9. Combination: alpha=0.2 + mc=2.0 + mr=None
        {**base, "alpha": 0.2, "max_correspondence_distance": 2.0,
         "map_radius": None, "tag": "9a_alpha0.2_mc2_mrNone"},
        # 10. Combination: alpha=0.0 + vox=0.3 + sv=0.15 + mc=1.5
        {**base, "alpha": 0.0, "voxel_size": 0.3, "source_voxel_size": 0.15,
         "max_correspondence_distance": 1.5, "tag": "10_alpha0_vox0.3"},
        # 11. alpha=0.5 + mf=50 (tight window)
        {**base, "max_map_frames": 50,  "tag": "11_mf50_base"},
        # 12. alpha=0.1 + max_map_frames=50 + map_radius=None
        {**base, "alpha": 0.1, "max_map_frames": 50, "map_radius": None,
         "tag": "12_alpha0.1_mf50_mrNone"},
    ]

    # ── Run all configs ───────────────────────────────────────────────────────
    results = []
    print(f"\n{'='*80}")
    print(f"  Grid search: {len(grid)} configs  seq{args.seq}  {n}fr  device={args.device}")
    print(f"  GT path={gt_path_len:.1f}m  KISS ATE={kiss_ate:.3f}m")
    print(f"{'='*80}")
    print(f"  {'Tag':<40} {'Path':>7} {'End':>6} {'ATE':>8} {'ms':>6} {'vs KISS':>8}")
    print(f"  {'-'*78}")

    for cfg_i, cfg in enumerate(grid):
        tag = cfg["tag"]
        print(f"  [{cfg_i+1:2d}/{len(grid)}] {tag:<38}", end="", flush=True)
        try:
            r = run_config(frames, timestamps, cfg, device=args.device, tmp_dir=tmp_dir)
            ate = compute_ate(gt_tum, r["tum_path"])
            delta = ""
            if ate is not None and kiss_ate is not None:
                pct = (ate - kiss_ate) / kiss_ate * 100
                delta = f"{pct:+.1f}%"
            ate_s = f"{ate:.3f}m" if ate is not None else "  N/A"
            print(f" {r['path_len']:>7.1f} {r['end_disp']:>6.2f} {ate_s:>8} "
                  f"{r['mean_ms']:>6.1f} {delta:>8}")
            results.append({"tag": tag, "config": cfg, "path_len": r["path_len"],
                             "end_disp": r["end_disp"], "ate": ate, "mean_ms": r["mean_ms"]})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"tag": tag, "config": cfg, "error": str(e)})

    # ── Summary: best configs ─────────────────────────────────────────────────
    valid = [r for r in results if "ate" in r and r["ate"] is not None]
    valid.sort(key=lambda x: x["ate"])

    print(f"\n{'='*80}")
    print(f"  SUMMARY  (seq{args.seq}, {n}fr)  GT path={gt_path_len:.1f}m  KISS={kiss_ate:.3f}m")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Tag':<40} {'ATE':>8} {'Path':>7} {'vs KISS':>8} {'ms':>6}")
    print(f"  {'-'*75}")
    for rank, r in enumerate(valid, 1):
        pct = (r["ate"] - kiss_ate) / kiss_ate * 100 if kiss_ate else 0
        marker = " <-- BEST" if rank == 1 else (" <-- KISS" if r["tag"] == "0_baseline" else "")
        print(f"  {rank:<5} {r['tag']:<40} {r['ate']:>8.3f}m {r['path_len']:>7.1f} "
              f"{pct:>+7.1f}% {r['mean_ms']:>6.1f}{marker}")

    # Save results
    out_json = tmp_dir / f"grid_search_seq{args.seq}_{n}fr.json"
    with open(out_json, "w") as f:
        json.dump({"seq": args.seq, "n_frames": n, "gt_path_m": gt_path_len,
                   "gt_end_disp_m": gt_end_disp,
                   "kiss_ate_m": kiss_ate, "kiss_path_m": kiss_path_len,
                   "results": results}, f, indent=2, default=str)
    print(f"\n  Results saved → {out_json}")


if __name__ == "__main__":
    main()
