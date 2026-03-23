"""
Metro Tunnel Grid Search - Phase 4: Final Validation
=======================================================
1) seq2 fr0-50 GT=0m 구간 분석 → IV-GICP가 fr0-50에서 폭발하는 이유 규명
2) Best config (sv0.25_mc1.0_itr12) 주변 fine-tune
3) 전체 3 시퀀스 최종 검증

Key findings from phases 1-3:
- Best single-seq: seq1=baseline_alpha0_mr60(0.471m), seq3=sv0.25_mc1.0_itr12(4.787m)
- Best cross-seq: sv0.25_mc1.0_itr12 (-26.2% vs KISS avg)
- seq2 is broken: GT has freeze in fr0-50, IV drift 67m (KISS drift 1.2m in same segment)
- seq2 GT JUMPS at fr77, 118, 145 suggest sparse/discontinuous RTK-GPS fix

Usage:
    uv run python examples/_metro_grid_search4.py --device cpu
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
    frame_disps = [np.linalg.norm(poses[i][:3,3] - poses[i-1][:3,3]) for i in range(1, len(poses))]
    return path_len, end_disp, float(np.mean(times)), tum_path, np.array(frame_disps)


def analyze_seq2_gt_structure(frames, max_frames=200):
    """seq2의 GT freezing 구조 분석 — RTK-GPS 지하 신호 손실?"""
    seq_gt = "Shield_tunnel2"
    gt_path = GEODE_ROOT / "groundtruth" / "metro_tunnel" / f"{seq_gt}.txt"
    data = np.loadtxt(gt_path)
    ts = data[:, 0]
    txyz = data[:, 1:4]
    quats = data[:, 4:8]

    print(f"\n  [seq2 RAW GT structure] total GT pts={len(ts)}")
    print(f"  GT time range: {ts[0]:.3f} ~ {ts[-1]:.3f}s  (span={ts[-1]-ts[0]:.1f}s)")

    # Check for duplicate/frozen positions
    pos_diffs = np.diff(txyz, axis=0)
    pos_norms = np.linalg.norm(pos_diffs, axis=1)
    frozen = np.where(pos_norms < 0.001)[0]
    print(f"  Frozen GT positions (diff<0.001m): {len(frozen)} points")
    if len(frozen) < 20:
        print(f"    At indices: {frozen}")

    # Check quat norms (zero quat = position-only)
    qnorms = np.linalg.norm(quats, axis=1)
    zero_q = np.where(qnorms < 0.1)[0]
    print(f"  Zero quaternions (position-only GT): {len(zero_q)}/{len(ts)} points")

    # Lidar frame timestamps
    lidar_ts = np.array([f[0] for f in frames[:max_frames]])
    print(f"\n  Lidar time range: {lidar_ts[0]:.3f} ~ {lidar_ts[-1]:.3f}s  "
          f"(span={lidar_ts[-1]-lidar_ts[0]:.1f}s)")

    # How well GT covers lidar timestamps?
    coverage_errors = []
    for lt in lidar_ts[:50]:
        idx = np.searchsorted(ts, lt)
        idx = np.clip(idx, 0, len(ts)-1)
        prev = max(0, idx-1)
        err = min(abs(ts[idx]-lt), abs(ts[prev]-lt))
        coverage_errors.append(err)
    print(f"  GT-Lidar timestamp matching error (fr0-50): "
          f"mean={np.mean(coverage_errors):.3f}s  max={np.max(coverage_errors):.3f}s")

    # Check if lidar ts[0-50] is BEFORE GT starts
    gt_start = ts[0]
    before_gt = np.sum(lidar_ts < gt_start)
    print(f"  Lidar frames BEFORE GT starts: {before_gt}")
    if before_gt > 0:
        print(f"  → IV-GICP has no valid GT for fr0-{before_gt}! "
              f"These frames use interpolated (=frozen) GT → ATE penalty")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    tmp_dir = Path(__file__).parent.parent / "results" / "geode" / "_gridsearch4"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── seq2 GT structure analysis ─────────────────────────────────────────────
    print("\n" + "═"*72)
    print("  SEQ2 GT STRUCTURE ANALYSIS")
    frames2 = None
    try:
        bag2 = GEODE_ROOT/"sensor_data"/"Metro_tunnel"/"Shield_tunnel2_gamma"/"Shield_tunnel2_gamma.bag"
        frames2 = read_metro_frames(bag2, max_frames=args.max_frames)
        analyze_seq2_gt_structure(frames2, max_frames=args.max_frames)
    except Exception as e:
        print(f"  Could not analyze seq2 GT: {e}")

    # ── Phase 4 configs: fine-tune around sv0.25_mc1.0_itr12 ─────────────────
    best_base = dict(voxel_size=0.5, source_voxel_size=0.25, alpha=0.0,
                     max_correspondence_distance=1.0, initial_threshold=1.5,
                     min_motion_th=0.5, max_map_frames=200, max_iterations=12,
                     map_radius=60.0)

    configs = {
        # Phase3 best
        "sv0.25_mc1.0_itr12_mr60":     {**best_base},
        "sv0.25_mc1.0_itr12_mrNone":   {**best_base, "map_radius": None},

        # Fine-tune mc around 1.0
        "sv0.25_mc0.8_itr12":          {**best_base, "max_correspondence_distance": 0.8},
        "sv0.25_mc1.2_itr12":          {**best_base, "max_correspondence_distance": 1.2},
        "sv0.25_mc1.5_itr12":          {**best_base, "max_correspondence_distance": 1.5},

        # Fine-tune sv around 0.25
        "sv0.22_mc1.0_itr12":          {**best_base, "source_voxel_size": 0.22},
        "sv0.28_mc1.0_itr12":          {**best_base, "source_voxel_size": 0.28},
        "sv0.30_mc1.0_itr12":          {**best_base, "source_voxel_size": 0.30},

        # itr variation around 12
        "sv0.25_mc1.0_itr10":          {**best_base, "max_iterations": 10},
        "sv0.25_mc1.0_itr15":          {**best_base, "max_iterations": 15},

        # min_motion_th on best base
        "sv0.25_mc1.0_mth0.3":         {**best_base, "min_motion_th": 0.3},
        "sv0.25_mc1.0_mth0.7":         {**best_base, "min_motion_th": 0.7},
        "sv0.25_mc1.0_mth1.0":         {**best_base, "min_motion_th": 1.0},

        # voxel size with mc=1.0
        "vox0.4_sv0.2_mc1.0_itr12":    {**best_base, "voxel_size": 0.4,
                                         "source_voxel_size": 0.2},
        "vox0.5_sv0.2_mc1.0_itr12":    {**best_base, "source_voxel_size": 0.2},
        "vox0.6_sv0.25_mc1.0_itr12":   {**best_base, "voxel_size": 0.6},

        # Combined: alpha=0.0 explicit + wider search
        "alpha0_sv0.25_mc0.8_itr12":   {**best_base, "alpha": 0.0,
                                         "max_correspondence_distance": 0.8},

        # Phase1 winner for seq1 (for reference comparison)
        "phase1_best_seq1":             dict(voxel_size=0.5, source_voxel_size=0.2, alpha=0.0,
                                             max_correspondence_distance=1.5, initial_threshold=1.5,
                                             min_motion_th=0.5, max_map_frames=200,
                                             max_iterations=20, map_radius=60.0),
    }

    all_results = {}

    for seq_num in ["1", "2", "3"]:
        seq_bag = f"Shield_tunnel{seq_num}_gamma"
        seq_gt  = f"Shield_tunnel{seq_num}"
        bag_path = GEODE_ROOT / "sensor_data" / "Metro_tunnel" / seq_bag / f"{seq_bag}.bag"
        gt_path  = GEODE_ROOT / "groundtruth" / "metro_tunnel" / f"{seq_gt}.txt"
        seq_tmp  = tmp_dir / f"seq{seq_num}"
        seq_tmp.mkdir(parents=True, exist_ok=True)

        print(f"\n{'═'*72}")
        print(f"  Seq {seq_num}")
        if seq_num == "2" and frames2 is not None:
            frames = frames2
            print(f"  (reusing loaded seq2 frames)")
        else:
            frames = read_metro_frames(bag_path, max_frames=args.max_frames)
        n = len(frames)
        timestamps = [f[0] for f in frames]

        gt_times, gt_poses = load_gt(gt_path)
        gt_at_lidar = interpolate_gt(gt_times, gt_poses, np.array(timestamps))
        gt_tum = seq_tmp / "gt.tum"
        save_tum(gt_at_lidar, timestamps, gt_tum)
        gt_path_len, gt_end_disp = traj_stats(gt_at_lidar)
        print(f"  GT path={gt_path_len:.1f}m  end={gt_end_disp:.1f}m  n={n}fr")

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

        results = {"KISS": {"ate": kiss_ate, "path_len": kiss_pl, "end_disp": kiss_ed,
                            "mean_ms": float(np.mean(kiss_times))}}

        print(f"\n  {'Tag':<42} {'Path':>7} {'End':>7} {'ATE':>8} {'ms':>6} {'vs KISS':>8}")
        print(f"  {'-'*78}")
        for tag, cfg_i in configs.items():
            try:
                pl, ed, ms, tum, fdisp = run_config(frames, timestamps, cfg_i,
                                                      args.device, seq_tmp, tag)
                ate = compute_ate(gt_tum, tum)
                pct = f"{(ate - kiss_ate)/kiss_ate*100:+.1f}%" if ate and kiss_ate else "N/A"
                ate_s = f"{ate:.3f}m" if ate else "N/A"
                print(f"  {tag:<42} {pl:>7.1f} {ed:>7.1f} {ate_s:>8} {ms:>6.1f} {pct:>8}")
                results[tag] = {"ate": ate, "path_len": pl, "end_disp": ed, "mean_ms": ms}
            except Exception as e:
                print(f"  {tag:<42} ERROR: {e}")

        all_results[f"seq{seq_num}"] = {
            "gt_path_m": gt_path_len, "gt_end_disp_m": gt_end_disp,
            "n_frames": n, "results": results
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PHASE 4 FINAL SUMMARY  {args.max_frames}fr/seq")
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
                np.mean(ates),
                ates
            )

    for tag, (pct, avg_ate, ates) in sorted(improvements.items(), key=lambda x: x[1][0]):
        ates_str = "  ".join([f"{a:.3f}" for a in ates])
        print(f"  {tag:<42} avg={avg_ate:.3f}m  {pct:+.1f}%  [{ates_str}]")

    kiss_avg = np.mean([all_results[sk]["results"]["KISS"]["ate"] for sk in all_results])
    kiss_ates = [all_results[sk]["results"]["KISS"]["ate"] for sk in all_results]
    print(f"\n  KISS avg ATE = {kiss_avg:.3f}m  [{' '.join(f'{a:.3f}' for a in kiss_ates)}]")

    # Best config recommendation
    best_tag = sorted(improvements.items(), key=lambda x: x[1][0])[0][0]
    best_pct = sorted(improvements.items(), key=lambda x: x[1][0])[0][1][0]
    print(f"\n  RECOMMENDATION: {best_tag}  ({best_pct:+.1f}% vs KISS)")
    best_cfg = configs[best_tag]
    print(f"    voxel={best_cfg['voxel_size']}  sv={best_cfg['source_voxel_size']}  "
          f"alpha={best_cfg['alpha']}  mc={best_cfg['max_correspondence_distance']}  "
          f"itr={best_cfg['max_iterations']}  map_radius={best_cfg.get('map_radius')}  "
          f"mf={best_cfg['max_map_frames']}  min_th={best_cfg['min_motion_th']}")

    out_json = tmp_dir / f"phase4_summary_{args.max_frames}fr.json"
    with open(out_json, "w") as f:
        json.dump({"results": all_results}, f, indent=2, default=str)
    print(f"  Results → {out_json}")


if __name__ == "__main__":
    main()
