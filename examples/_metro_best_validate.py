"""
Metro Tunnel Best Config Validation - 500fr
=============================================
Validates the best config found in grid search:
  voxel=0.5, sv=0.25, alpha=0.0, mc=1.0, itr=12, mr=60, mf=200, min_th=0.5

Runs all 3 sequences with original (baseline) AND best config side-by-side.

Usage:
    uv run python examples/_metro_best_validate.py --device cpu
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
    pts  = np.stack([x, y, z, refl], axis=1).astype(np.float64)
    r    = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    return secs * 1e9 + nsecs, pts[valid]


def read_metro_frames(bag_path: Path, max_frames=None, max_range=60.0):
    from rosbags.rosbag1 import Reader
    frames = []
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        total = conns[0].msgcount if conns else 0
        for conn, ts_ns, raw in bag.messages(connections=conns):
            try:
                t_ns, pts = parse_livox_frame(raw, max_range=max_range)
                if pts is not None and len(pts) > 100:
                    frames.append((t_ns / 1e9, pts))
                    if len(frames) % 200 == 0:
                        print(f"  {len(frames)}/{total}...", end="\r")
                    if max_frames and len(frames) >= max_frames:
                        break
            except Exception:
                continue
    print(f"  Loaded {len(frames)} frames  avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame")
    return frames


def load_gt(gt_path):
    from scipy.spatial.transform import Rotation
    data = np.loadtxt(gt_path)
    ts, txyz, quats = data[:, 0], data[:, 1:4], data[:, 4:8]
    T0_inv, poses = None, []
    for i in range(len(ts)):
        q = quats[i]
        R = Rotation.from_quat(q).as_matrix() if np.linalg.norm(q) > 1e-6 else np.eye(3)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = txyz[i]
        if T0_inv is None: T0_inv = np.linalg.inv(T)
        poses.append(T0_inv @ T)
    return ts, np.stack(poses)


def interpolate_gt(gt_times, gt_poses, lidar_times):
    idx  = np.searchsorted(gt_times, lidar_times)
    idx  = np.clip(idx, 0, len(gt_times) - 1)
    prev = np.clip(idx - 1, 0, len(gt_times) - 1)
    best = np.where(np.abs(gt_times[prev]-lidar_times) < np.abs(gt_times[idx]-lidar_times), prev, idx)
    return gt_poses[best]


def save_tum(poses, timestamps, path):
    from scipy.spatial.transform import Rotation
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def traj_stats(poses):
    d = [np.linalg.norm(poses[i+1][:3,3]-poses[i][:3,3]) for i in range(len(poses)-1)]
    return float(np.sum(d)), float(np.linalg.norm(poses[-1][:3,3]-poses[0][:3,3]))


def compose_poses(rel_list):
    poses = [np.eye(4)]
    for T in rel_list: poses.append(poses[-1] @ T)
    return poses


def compute_ate(gt_tum, pred_tum):
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
    except Exception:
        return None


def run_iv_config(frames, timestamps, cfg, device, tmp_dir, tag):
    from iv_gicp.pipeline import IVGICPPipeline
    pipeline = IVGICPPipeline(
        voxel_size=cfg["voxel_size"],
        source_voxel_size=cfg["source_voxel_size"],
        alpha=cfg["alpha"],
        max_correspondence_distance=cfg["max_correspondence_distance"],
        initial_threshold=cfg.get("initial_threshold", 1.5),
        min_motion_th=cfg.get("min_motion_th", 0.5),
        max_map_frames=cfg.get("max_map_frames", 200),
        max_iterations=cfg.get("max_iterations", 20),
        map_radius=cfg.get("map_radius", 60.0),
        auto_alpha=False, auto_alpha_from_intensity=False,
        source_drop_small_voxels=False, source_max_output_features=0,
        source_min_feature_score=0.0, max_source_points=0,
        device=device,
    )
    abs_poses = [np.eye(4)]; times = []
    for ts, pts in frames:
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        abs_poses.append(result.pose.copy())
    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    poses = compose_poses(rel)
    tum = Path(tmp_dir) / f"{tag}.tum"
    save_tum(poses, timestamps, tum)
    return poses, float(np.mean(times)), tum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "results" / "geode" / "_best_validate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configs to compare
    orig_cfg = dict(voxel_size=0.5, source_voxel_size=0.2, alpha=0.5,
                    max_correspondence_distance=1.5, initial_threshold=1.5,
                    min_motion_th=0.5, max_map_frames=200, max_iterations=20,
                    map_radius=60.0)

    best_cfg = dict(voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
                    max_correspondence_distance=1.0, initial_threshold=1.5,
                    min_motion_th=0.5, max_map_frames=200, max_iterations=12,
                    map_radius=60.0)

    # Also test mc=0.8 (was best on seq2)
    mc08_cfg = dict(voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
                    max_correspondence_distance=0.8, initial_threshold=1.5,
                    min_motion_th=0.5, max_map_frames=200, max_iterations=12,
                    map_radius=60.0)

    summary = {}

    for seq_num in ["1", "2", "3"]:
        seq_bag = f"Shield_tunnel{seq_num}_gamma"
        bag_path = GEODE_ROOT/"sensor_data"/"Metro_tunnel"/seq_bag/f"{seq_bag}.bag"
        gt_path  = GEODE_ROOT/"groundtruth"/"metro_tunnel"/f"Shield_tunnel{seq_num}.txt"
        seq_tmp  = out_dir / f"seq{seq_num}"
        seq_tmp.mkdir(parents=True, exist_ok=True)

        print(f"\n{'═'*72}")
        print(f"  Seq {seq_num}  ({args.max_frames}fr)")
        frames = read_metro_frames(bag_path, max_frames=args.max_frames)
        n = len(frames)
        timestamps = [f[0] for f in frames]

        gt_times, gt_poses = load_gt(gt_path)
        gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
        gt_tum = seq_tmp / "gt.tum"
        save_tum(gt_at_lidar, timestamps, gt_tum)
        gt_pl, gt_ed = traj_stats(gt_at_lidar)
        print(f"  GT: path={gt_pl:.1f}m  end={gt_ed:.1f}m")

        # KISS
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.config import KISSConfig
        kc = KISSConfig()
        kc.data.deskew = False; kc.data.max_range = 60.0; kc.data.min_range = 0.5
        kc.mapping.voxel_size = 0.8
        od = KissICP(config=kc)
        kiss_poses = [np.eye(4)]; kiss_times = []
        for ts, pts in frames:
            t0 = time.perf_counter()
            od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
            kiss_times.append((time.perf_counter() - t0) * 1000)
            kiss_poses.append(od.last_pose.copy())
        kiss_tum = seq_tmp / "kiss.tum"
        save_tum(kiss_poses, timestamps, kiss_tum)
        kiss_pl2, kiss_ed2 = traj_stats(kiss_poses)
        kiss_ate = compute_ate(gt_tum, kiss_tum)

        # IV orig
        iv_orig_poses, iv_orig_ms, iv_orig_tum = run_iv_config(
            frames, timestamps, orig_cfg, args.device, seq_tmp, "iv_orig")
        orig_pl, orig_ed = traj_stats(iv_orig_poses)
        orig_ate = compute_ate(gt_tum, iv_orig_tum)

        # IV best
        iv_best_poses, iv_best_ms, iv_best_tum = run_iv_config(
            frames, timestamps, best_cfg, args.device, seq_tmp, "iv_best")
        best_pl, best_ed = traj_stats(iv_best_poses)
        best_ate = compute_ate(gt_tum, iv_best_tum)

        # IV mc=0.8
        iv_mc08_poses, iv_mc08_ms, iv_mc08_tum = run_iv_config(
            frames, timestamps, mc08_cfg, args.device, seq_tmp, "iv_mc08")
        mc08_pl, mc08_ed = traj_stats(iv_mc08_poses)
        mc08_ate = compute_ate(gt_tum, iv_mc08_tum)

        print(f"\n  {'Method':<35} {'Path':>7} {'End':>7} {'ATE':>9} {'ms':>6} {'vs KISS':>9}")
        print(f"  {'-'*76}")
        print(f"  {'GT':<35} {gt_pl:>7.1f} {gt_ed:>7.1f}")
        print(f"  {'KISS-ICP':<35} {kiss_pl2:>7.1f} {kiss_ed2:>7.1f} {kiss_ate:>9.3f}m {np.mean(kiss_times):>6.1f}")

        def fmt_row(name, pl, ed, ate, ms):
            pct = f"{(ate-kiss_ate)/kiss_ate*100:+.1f}%" if ate and kiss_ate else "N/A"
            marker = " ✓" if ate and kiss_ate and ate < kiss_ate else ""
            print(f"  {name:<35} {pl:>7.1f} {ed:>7.1f} {ate:>9.3f}m {ms:>6.1f} {pct:>9}{marker}")

        fmt_row("IV orig (α=0.5,sv=0.2,mc=1.5,itr=20)", orig_pl, orig_ed, orig_ate, iv_orig_ms)
        fmt_row("IV best (α=0.0,sv=0.25,mc=1.0,itr=12)", best_pl, best_ed, best_ate, iv_best_ms)
        fmt_row("IV mc0.8 (α=0.0,sv=0.25,mc=0.8,itr=12)", mc08_pl, mc08_ed, mc08_ate, iv_mc08_ms)

        summary[f"seq{seq_num}"] = {
            "gt_path_m": gt_pl, "n_frames": n,
            "KISS": {"ate": kiss_ate, "path_len": kiss_pl2, "mean_ms": float(np.mean(kiss_times))},
            "IV_orig": {"ate": orig_ate, "path_len": orig_pl, "mean_ms": iv_orig_ms},
            "IV_best": {"ate": best_ate, "path_len": best_pl, "mean_ms": iv_best_ms},
            "IV_mc08": {"ate": mc08_ate, "path_len": mc08_pl, "mean_ms": iv_mc08_ms},
        }

    # Overall summary
    print(f"\n{'='*80}")
    print(f"  OVERALL SUMMARY  {args.max_frames}fr/seq  ({3} sequences)")
    print(f"{'='*80}")
    for method in ["KISS", "IV_orig", "IV_best", "IV_mc08"]:
        ates  = [summary[sk][method]["ate"] for sk in summary if summary[sk][method].get("ate")]
        kis_a = [summary[sk]["KISS"]["ate"] for sk in summary if summary[sk]["KISS"].get("ate")]
        avg_ate = np.mean(ates) if ates else float("nan")
        avg_pct = np.mean([(a-k)/k*100 for a,k in zip(ates, kis_a)]) if method!="KISS" and ates else 0
        ates_str = "  ".join([f"{a:.3f}" for a in ates])
        marker = f"  {avg_pct:+.1f}% vs KISS" if method != "KISS" else ""
        print(f"  {method:<12} avg={avg_ate:.3f}m  [{ates_str}]{marker}")

    out_json = out_dir / f"best_validate_{args.max_frames}fr.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results → {out_json}")


if __name__ == "__main__":
    main()
