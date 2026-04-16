"""
Metro Tunnel Grid Search - Phase 2
====================================
1) Top configs on seq1 + seq2 (폭발 원인 분석)
2) alpha=0.0 / map_radius 조합 집중 탐색

Usage:
    uv run python examples/_metro_grid_search2.py --device cpu
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

GEODE_ROOT = Path("/home/km/deepai_dev_data/GEODE")
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
    for T in rel_list:
        poses.append(poses[-1] @ T)
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
        print(f"  evo failed: {e}")
        return None


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
    abs_poses = [np.eye(4)]
    times = []
    for ts, pts in frames:
        t0 = time.perf_counter()
        result = pipeline.process_frame(pts[:, :3], pts[:, 3], timestamp=ts)
        times.append((time.perf_counter() - t0) * 1000)
        abs_poses.append(result.pose.copy())

    rel = [np.linalg.inv(abs_poses[i-1]) @ abs_poses[i] for i in range(1, len(abs_poses))]
    poses = compose_poses(rel)
    tum_path = tmp_dir / f"pred_{tag}.tum"
    save_tum(poses, timestamps, tum_path)
    path_len, end_disp = traj_stats(poses)
    return path_len, end_disp, float(np.mean(times)), tum_path


def run_sequence(seq_num, configs, max_frames, device, tmp_dir):
    seq_bag = f"Shield_tunnel{seq_num}_gamma"
    seq_gt  = f"Shield_tunnel{seq_num}"
    bag_path = GEODE_ROOT / "sensor_data" / "Metro_tunnel" / seq_bag / f"{seq_bag}.bag"
    gt_path  = GEODE_ROOT / "groundtruth" / "metro_tunnel" / f"{seq_gt}.txt"
    seq_tmp  = tmp_dir / f"seq{seq_num}"
    seq_tmp.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"  Seq {seq_num}: {bag_path.name}")
    frames = read_metro_frames(bag_path, max_frames=max_frames)
    n = len(frames)
    timestamps = [f[0] for f in frames]

    gt_times, gt_poses = load_gt(gt_path)
    gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
    gt_tum = seq_tmp / "gt.tum"
    save_tum(gt_at_lidar, timestamps, gt_tum)
    gt_path_len, gt_end_disp = traj_stats(gt_at_lidar)
    print(f"  GT path={gt_path_len:.1f}m  end={gt_end_disp:.1f}m  n={n}fr")

    # KISS baseline
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
    print(f"  KISS: path={kiss_pl:.1f}m end={kiss_ed:.1f}m ATE={kiss_ate:.3f}m "
          f"mean={np.mean(kiss_times):.1f}ms")

    results = {"KISS": {"ate": kiss_ate, "path_len": kiss_pl, "end_disp": kiss_ed,
                        "mean_ms": float(np.mean(kiss_times))}}

    print(f"\n  {'Tag':<38} {'Path':>7} {'End':>7} {'ATE':>8} {'ms':>6} {'vs KISS':>8}")
    print(f"  {'-'*72}")
    for tag, cfg_i in configs.items():
        try:
            pl, ed, ms, tum = run_config(frames, timestamps, cfg_i, device, seq_tmp, tag)
            ate = compute_ate(gt_tum, tum)
            pct = f"{(ate - kiss_ate)/kiss_ate*100:+.1f}%" if ate and kiss_ate else "  N/A"
            ate_s = f"{ate:.3f}m" if ate else "   N/A"
            print(f"  {tag:<38} {pl:>7.1f} {ed:>7.1f} {ate_s:>8} {ms:>6.1f} {pct:>8}")
            results[tag] = {"ate": ate, "path_len": pl, "end_disp": ed, "mean_ms": ms}
        except Exception as e:
            print(f"  {tag:<38} ERROR: {e}")

    return {"gt_path_m": gt_path_len, "gt_end_disp_m": gt_end_disp,
            "n_frames": n, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seq",        default="all", choices=["1","2","3","all"])
    args = parser.parse_args()

    tmp_dir = Path(__file__).parent.parent / "results" / "geode" / "_gridsearch2"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: focused configs based on phase 1 insights
    # Key finding: alpha=0.0, map_radius>=80 (or None), sv=0.2 are best
    base_good = dict(voxel_size=0.5, source_voxel_size=0.2, alpha=0.0,
                     max_correspondence_distance=1.5, initial_threshold=1.5,
                     min_motion_th=0.5, max_map_frames=200, max_iterations=20,
                     map_radius=None)

    configs = {
        # Phase 1 top configs
        "A_alpha0_mrNone":        {**base_good},
        "B_alpha0_mr60":          {**base_good, "map_radius": 60.0},
        "C_alpha0_mr80":          {**base_good, "map_radius": 80.0},
        "D_alpha0.5_mr60_orig":   {**base_good, "alpha": 0.5, "map_radius": 60.0},

        # Fine-tune mc with alpha=0.0
        "E_alpha0_mc1.0_mrNone":  {**base_good, "max_correspondence_distance": 1.0},
        "F_alpha0_mc2.0_mrNone":  {**base_good, "max_correspondence_distance": 2.0},

        # max_range variation (Livox 60m default, test shorter/longer)
        "G_alpha0_range30":       {**base_good,
                                   # we pass via pipeline; needs special handling below
                                   "max_range_override": 30.0},
        "H_alpha0_range80":       {**base_good, "max_range_override": 80.0},

        # sv fine-tune
        "I_alpha0_sv0.15":        {**base_good, "source_voxel_size": 0.15},
        "J_alpha0_sv0.25":        {**base_good, "source_voxel_size": 0.25},

        # mf variation
        "K_alpha0_mf100":         {**base_good, "max_map_frames": 100},
        "L_alpha0_mf50":          {**base_good, "max_map_frames": 50},

        # itr variation
        "M_alpha0_itr12":         {**base_good, "max_iterations": 12},
        "N_alpha0_itr30":         {**base_good, "max_iterations": 30},

        # min_motion_th — seq2 path explosion may be due to too-small threshold
        "O_alpha0_mth1.0":        {**base_good, "min_motion_th": 1.0},
        "P_alpha0_mth2.0":        {**base_good, "min_motion_th": 2.0},

        # Best combo candidates
        "Q_alpha0_mc1_sv0.2_mrNone": {**base_good, "max_correspondence_distance": 1.0,
                                       "min_motion_th": 0.5},
        "R_alpha0_mth1_mc1.5":       {**base_good, "min_motion_th": 1.0,
                                       "max_correspondence_distance": 1.5},
        "S_alpha0_mth2_mc2":         {**base_good, "min_motion_th": 2.0,
                                       "max_correspondence_distance": 2.0},
    }

    # Filter out max_range_override (requires custom run_config); handle separately
    # For simplicity, skip range tests here as pipeline has fixed range from data loader

    seqs = ["1", "2", "3"] if args.seq == "all" else [args.seq]
    all_results = {}

    for seq_num in seqs:
        # Strip max_range_override from configs (not supported in run_config)
        clean_configs = {k: {kk: vv for kk, vv in v.items() if kk != "max_range_override"}
                         for k, v in configs.items()
                         if not v.get("max_range_override")}
        res = run_sequence(seq_num, clean_configs, args.max_frames, args.device, tmp_dir)
        all_results[f"seq{seq_num}"] = res

    # ── Cross-sequence summary ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  CROSS-SEQUENCE SUMMARY  ({args.max_frames}fr/seq)")
    print(f"{'='*80}")

    # Collect all method names
    method_names = ["KISS"]
    for seq_data in all_results.values():
        for m in seq_data["results"].keys():
            if m not in method_names:
                method_names.append(m)

    # Header
    seq_keys = list(all_results.keys())
    hdr = f"  {'Method':<38}"
    for sk in seq_keys:
        hdr += f" {'ATE_'+sk:>12}"
    hdr += f"  {'avg_vs_KISS':>12}"
    print(hdr)
    print(f"  {'-'*78}")

    # Per method
    method_rows = {}
    for m in method_names:
        ates = []
        kiss_ates = []
        row = f"  {m:<38}"
        for sk in seq_keys:
            res = all_results[sk]["results"]
            if m in res and res[m].get("ate") is not None:
                ate = res[m]["ate"]
                ates.append(ate)
                row += f" {ate:>12.3f}"
            else:
                row += f" {'N/A':>12}"
            kiss_ate = res.get("KISS", {}).get("ate")
            if kiss_ate:
                kiss_ates.append(kiss_ate)

        if ates and kiss_ates and m != "KISS":
            avg_pct = np.mean([(a - k) / k * 100 for a, k in zip(ates, kiss_ates)])
            row += f"  {avg_pct:>+11.1f}%"
        print(row)
        method_rows[m] = ates

    # Find best
    print(f"\n  Best configs by avg ATE improvement vs KISS:")
    improvements = {}
    for m in method_names:
        if m == "KISS":
            continue
        ates = []
        kiss_ates = []
        for sk in seq_keys:
            res = all_results[sk]["results"]
            if m in res and res[m].get("ate"):
                ates.append(res[m]["ate"])
                kiss_ates.append(res["KISS"]["ate"])
        if ates and kiss_ates:
            improvements[m] = np.mean([(a-k)/k*100 for a, k in zip(ates, kiss_ates)])

    for m, pct in sorted(improvements.items(), key=lambda x: x[1])[:10]:
        print(f"    {m:<40} {pct:+.1f}%")

    out_json = tmp_dir / f"phase2_summary_{args.max_frames}fr.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results → {out_json}")


if __name__ == "__main__":
    main()
