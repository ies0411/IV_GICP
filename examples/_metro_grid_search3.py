"""
Metro Tunnel Grid Search - Phase 3
=====================================
1) seq2 path explosion 원인 분석: per-frame displacement vs GT 비교
2) 상위 config 조합 탐색 (sv=0.25, itr=12 포함)
3) seq2에 특화된 파라미터 탐색

Usage:
    uv run python examples/_metro_grid_search3.py --device cpu
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
    pts = np.stack([x, y, z, refl], axis=1).astype(np.float64)
    r   = np.linalg.norm(pts[:, :3], axis=1)
    valid = np.isfinite(pts).all(axis=1) & (r > 0.5) & (r < max_range)
    ts_ns = secs * 1e9 + nsecs
    return ts_ns, pts[valid]


def read_metro_frames(bag_path: Path, max_frames: int = None, max_range: float = 60.0):
    from rosbags.rosbag1 import Reader
    frames = []
    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == LIDAR_TOPIC]
        for conn, ts_ns, raw in bag.messages(connections=conns):
            try:
                t_ns, pts = parse_livox_frame(raw, max_range=max_range)
                if pts is not None and len(pts) > 100:
                    frames.append((t_ns / 1e9, pts))
                    if max_frames and len(frames) >= max_frames:
                        break
            except Exception:
                continue
    print(f"  Loaded {len(frames)} frames  avg {np.mean([len(f[1]) for f in frames]):.0f} pts/frame")
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
        R = Rotation.from_quat(q).as_matrix() if np.linalg.norm(q) > 1e-6 else np.eye(3)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = txyz[i]
        if T0_inv is None: T0_inv = np.linalg.inv(T)
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
    except Exception as e:
        return None


def analyze_seq2_displacement(frames, gt_at_lidar):
    """Per-frame GT displacement statistics to understand seq2 motion pattern."""
    print("\n  [seq2 GT analysis] Per-segment displacement stats:")
    n = len(gt_at_lidar)
    gt_disp_per_frame = [np.linalg.norm(gt_at_lidar[i][:3,3] - gt_at_lidar[i-1][:3,3])
                         for i in range(1, n)]
    gt_disp = np.array(gt_disp_per_frame)

    # Check for jumps (possible GT quality issues)
    jump_threshold = 1.0  # 1m per frame = 10 m/s at 10Hz → unreasonable
    jumps = np.where(gt_disp > jump_threshold)[0]
    if len(jumps) > 0:
        print(f"    GT JUMPS (>{jump_threshold}m/frame) at frames: {jumps[:20]}")
        for j in jumps[:10]:
            print(f"      frame {j}: disp={gt_disp[j]:.3f}m  "
                  f"t=[{gt_at_lidar[j][:3,3]}] -> [{gt_at_lidar[j+1][:3,3]}]")
    else:
        print(f"    No GT jumps found (max={gt_disp.max():.3f}m/frame)")

    # Segments stats
    for seg_start, seg_end, label in [(0, 50, "fr0-50"),
                                       (50, 100, "fr50-100"),
                                       (100, 150, "fr100-150"),
                                       (150, min(200, n-1), "fr150-200")]:
        seg = gt_disp[seg_start:seg_end]
        if len(seg) > 0:
            print(f"    {label}: mean={seg.mean():.3f}m  max={seg.max():.3f}m  "
                  f"sum={seg.sum():.1f}m  n_jumps={np.sum(seg>0.5)}")

    # Check if GT timestamps are uniform
    lidar_ts = np.array([f[0] for f in frames[:n]])
    ts_gaps = np.diff(lidar_ts)
    print(f"\n    Lidar timestamp gaps: mean={ts_gaps.mean():.3f}s  "
          f"max={ts_gaps.max():.3f}s  std={ts_gaps.std():.4f}s")
    big_gaps = np.where(ts_gaps > 0.2)[0]
    if len(big_gaps) > 0:
        print(f"    BIG GAPS (>0.2s) at: {big_gaps}")

    return gt_disp


def run_config(frames, timestamps, config, device, tmp_dir, tag):
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
    abs_poses = [np.eye(4)]; times = []
    # Per-frame displacement
    frame_disps = []
    prev_pose = np.eye(4)
    for ts, pts in frames:
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        abs_poses.append(result.pose.copy())
        disp = np.linalg.norm(result.pose[:3, 3] - prev_pose[:3, 3])
        frame_disps.append(disp)
        prev_pose = result.pose.copy()

    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    poses = compose_poses(rel)
    tum_path = tmp_dir / f"pred_{tag}.tum"
    save_tum(poses, timestamps, tum_path)
    path_len, end_disp = traj_stats(poses)
    return path_len, end_disp, float(np.mean(times)), tum_path, np.array(frame_disps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    tmp_dir = Path(__file__).parent.parent / "results" / "geode" / "_gridsearch3"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Phase 3 configs: combinations of best phase 1+2 insights
    # Key: sv=0.25, itr=12, mr=60 each show partial wins
    base = dict(voxel_size=0.5, source_voxel_size=0.2, alpha=0.0,
                max_correspondence_distance=1.5, initial_threshold=1.5,
                min_motion_th=0.5, max_map_frames=200, max_iterations=20,
                map_radius=60.0)

    configs = {
        # Current best baseline (phase 1 winner overall: alpha=0, mr=60)
        "baseline_alpha0_mr60":      {**base},

        # sv=0.25 was best in cross-seq avg
        "sv0.25_mr60":               {**base, "source_voxel_size": 0.25},
        "sv0.25_mrNone":             {**base, "source_voxel_size": 0.25, "map_radius": None},
        "sv0.25_itr12":              {**base, "source_voxel_size": 0.25, "max_iterations": 12},
        "sv0.25_itr12_mr60":         {**base, "source_voxel_size": 0.25, "max_iterations": 12},
        "sv0.25_itr12_mrNone":       {**base, "source_voxel_size": 0.25, "max_iterations": 12,
                                      "map_radius": None},

        # itr=12 was 2nd best overall
        "itr12_mr60":                {**base, "max_iterations": 12},
        "itr12_mrNone":              {**base, "max_iterations": 12, "map_radius": None},

        # mc variations with best alpha/sv combos
        "sv0.25_mc1.0_itr12":        {**base, "source_voxel_size": 0.25,
                                      "max_correspondence_distance": 1.0, "max_iterations": 12},
        "sv0.25_mc2.0_itr12":        {**base, "source_voxel_size": 0.25,
                                      "max_correspondence_distance": 2.0, "max_iterations": 12},

        # voxel_size with sv=0.25
        "vox0.4_sv0.2_itr12":        {**base, "voxel_size": 0.4, "max_iterations": 12},
        "vox0.6_sv0.25_itr12":       {**base, "voxel_size": 0.6, "source_voxel_size": 0.25,
                                      "max_iterations": 12},
        "vox0.7_sv0.3_itr12":        {**base, "voxel_size": 0.7, "source_voxel_size": 0.3,
                                      "max_iterations": 12},

        # mf=100 moderate window + sv=0.25
        "sv0.25_mf100_itr12":        {**base, "source_voxel_size": 0.25, "max_map_frames": 100,
                                      "max_iterations": 12},
        "sv0.25_mf100_mr60":         {**base, "source_voxel_size": 0.25, "max_map_frames": 100},

        # Try alpha=0.1 (small intensity contribution)
        "alpha0.1_sv0.25_itr12":     {**base, "alpha": 0.1, "source_voxel_size": 0.25,
                                      "max_iterations": 12},

        # seq2 specific: what if the issue is motion prediction outlier? Try larger mc
        "sv0.25_mc3.0_itr12":        {**base, "source_voxel_size": 0.25,
                                      "max_correspondence_distance": 3.0, "max_iterations": 12},

        # Original paper config for reference
        "original_alpha0.5_mr60_itr20": {**base, "alpha": 0.5},
    }

    all_results = {}
    all_frame_disps = {}

    for seq_num in ["1", "2", "3"]:
        seq_bag = f"Shield_tunnel{seq_num}_gamma"
        seq_gt  = f"Shield_tunnel{seq_num}"
        bag_path = GEODE_ROOT / "sensor_data" / "Metro_tunnel" / seq_bag / f"{seq_bag}.bag"
        gt_path  = GEODE_ROOT / "groundtruth" / "metro_tunnel" / f"{seq_gt}.txt"
        seq_tmp  = tmp_dir / f"seq{seq_num}"
        seq_tmp.mkdir(parents=True, exist_ok=True)

        print(f"\n{'═'*72}")
        print(f"  Seq {seq_num}")
        frames = read_metro_frames(bag_path, max_frames=args.max_frames)
        n = len(frames)
        timestamps = [f[0] for f in frames]

        gt_times, gt_poses = load_gt(gt_path)
        gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
        gt_tum = seq_tmp / "gt.tum"
        save_tum(gt_at_lidar, timestamps, gt_tum)
        gt_path_len, gt_end_disp = traj_stats(gt_at_lidar)
        print(f"  GT path={gt_path_len:.1f}m  end={gt_end_disp:.1f}m  n={n}fr")

        # GT analysis for seq2
        if seq_num == "2":
            gt_disp = analyze_seq2_displacement(frames, gt_at_lidar)

        # KISS
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.config import KISSConfig
        cfg = KISSConfig()
        cfg.data.deskew = False; cfg.data.max_range = 60.0; cfg.data.min_range = 0.5
        cfg.mapping.voxel_size = 0.8
        od = KissICP(config=cfg)
        kiss_poses = [np.eye(4)]; kiss_times = []
        for ts, pts in frames:
            t0 = time.perf_counter()
            od.register_frame(pts[:, :3].astype(np.float64), np.full(len(pts), ts))
            kiss_times.append((time.perf_counter() - t0) * 1000)
            kiss_poses.append(od.last_pose.copy())
        kiss_tum = seq_tmp / "kiss.tum"
        save_tum(kiss_poses, timestamps, kiss_tum)
        kiss_pl, kiss_ed = traj_stats(kiss_poses)
        kiss_ate = compute_ate(gt_tum, kiss_tum)
        print(f"  KISS: path={kiss_pl:.1f}m  end={kiss_ed:.1f}m  ATE={kiss_ate:.3f}m "
              f"({np.mean(kiss_times):.1f}ms)")

        kiss_frame_disps = [np.linalg.norm(kiss_poses[i][:3,3] - kiss_poses[i-1][:3,3])
                            for i in range(1, len(kiss_poses))]
        if seq_num == "2":
            kiss_disp_arr = np.array(kiss_frame_disps)
            gt_disp_arr = np.array([np.linalg.norm(gt_at_lidar[i][:3,3]-gt_at_lidar[i-1][:3,3])
                                    for i in range(1, len(gt_at_lidar))])
            print(f"\n  [seq2 KISS vs GT disp]")
            for seg_s, seg_e, lbl in [(0,50,"fr0-50"),(50,100,"fr50-100"),(100,150,"fr100-150"),(150,min(200,n-1),"fr150-200")]:
                kd = kiss_disp_arr[seg_s:seg_e]
                gd = gt_disp_arr[seg_s:seg_e]
                if len(kd)>0 and len(gd)>0:
                    print(f"    {lbl}: KISS={kd.mean():.3f}m/fr (sum={kd.sum():.1f}m)  "
                          f"GT={gd.mean():.3f}m/fr (sum={gd.sum():.1f}m)  "
                          f"ratio={kd.sum()/max(gd.sum(),0.001):.2f}")

        results = {"KISS": {"ate": kiss_ate, "path_len": kiss_pl, "end_disp": kiss_ed,
                            "mean_ms": float(np.mean(kiss_times))}}

        print(f"\n  {'Tag':<42} {'Path':>7} {'End':>7} {'ATE':>8} {'ms':>6} {'vs KISS':>8}")
        print(f"  {'-'*76}")
        for tag, cfg_i in configs.items():
            try:
                pl, ed, ms, tum, fdisp = run_config(frames, timestamps, cfg_i,
                                                      args.device, seq_tmp, tag)
                ate = compute_ate(gt_tum, tum)
                pct = f"{(ate - kiss_ate)/kiss_ate*100:+.1f}%" if ate and kiss_ate else "N/A"
                ate_s = f"{ate:.3f}m" if ate else "N/A"
                print(f"  {tag:<42} {pl:>7.1f} {ed:>7.1f} {ate_s:>8} {ms:>6.1f} {pct:>8}")
                results[tag] = {"ate": ate, "path_len": pl, "end_disp": ed, "mean_ms": ms}
                all_frame_disps[f"seq{seq_num}_{tag}"] = fdisp.tolist()
            except Exception as e:
                print(f"  {tag:<42} ERROR: {e}")

        all_results[f"seq{seq_num}"] = {
            "gt_path_m": gt_path_len, "gt_end_disp_m": gt_end_disp,
            "n_frames": n, "results": results
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY  Phase3  {args.max_frames}fr/seq")
    print(f"{'='*80}")
    improvements = {}
    for tag in configs.keys():
        ates, kiss_ates = [], []
        for sk in all_results.keys():
            res = all_results[sk]["results"]
            if tag in res and res[tag].get("ate"):
                ates.append(res[tag]["ate"])
                kiss_ates.append(res["KISS"]["ate"])
        if ates and kiss_ates:
            improvements[tag] = (
                np.mean([(a-k)/k*100 for a, k in zip(ates, kiss_ates)]),
                np.mean(ates)
            )

    # Sort by avg improvement
    for tag, (pct, avg_ate) in sorted(improvements.items(), key=lambda x: x[1][0]):
        ates_str = "  ".join([f"{all_results[sk]['results'].get(tag, {}).get('ate', float('nan')):.3f}"
                               for sk in all_results])
        print(f"  {tag:<42} avg={avg_ate:.3f}m  {pct:+.1f}%  [{ates_str}]")

    kiss_avg = np.mean([all_results[sk]["results"]["KISS"]["ate"] for sk in all_results])
    print(f"\n  KISS avg ATE = {kiss_avg:.3f}m")

    out_json = tmp_dir / f"phase3_summary_{args.max_frames}fr.json"
    with open(out_json, "w") as f:
        json.dump({"results": all_results}, f, indent=2, default=str)
    print(f"  Results → {out_json}")


if __name__ == "__main__":
    main()
